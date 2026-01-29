import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# 导入依赖模块
from llm_hidden_extractor import extract_hidden_states
from m_t_calculator import compute_m_t_full, ChineseMorphExtractor
from c_t_calculator import compute_c_t
from d_t_calculator import compute_d_t


class SemanticDriftCoeff(nn.Module):
    """语义漂移系数S(t)的PyTorch实现

    公式: S(t) = [dist(h(t), h_global(t)) / dist_max(t)] * [1 - ω·C(t) + ν·D(t)] * ξ(M(t))
    """

    def __init__(self,
                 lambda_decay: float = 0.1,
                 middle_layer_idx: int = 12,
                 normalize_hidden: bool = True,
                 use_global_anchor: bool = False):
        """
        Args:
            lambda_decay: 时间衰减系数λ，默认0.1
            middle_layer_idx: 中间层索引，默认12
            normalize_hidden: 是否对hidden states做L2归一化，消除首Token范数异常影响
            use_global_anchor: 是否使用全局句子级锚点（需双遍历），默认False使用动态锚点
        """
        super(SemanticDriftCoeff, self).__init__()
        # 可训练参数μ，初始化为0.25
        self.mu = nn.Parameter(torch.tensor(0.25))
        # 固定参数
        self.omega = 0.3
        self.nu = 0.2
        self.lambda_decay = lambda_decay
        self.normalize_hidden = normalize_hidden
        self.use_global_anchor = use_global_anchor

        # LLM相关参数（固定值）
        self.model_name = "D:\\liubotao\\other\\BIT_TS\\LLM_GCG\\code\\models\\Qwen2___5-0___5B-Instruct"
        self.middle_layer_idx = middle_layer_idx
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 初始化形态特征提取器
        self.morph_extractor = ChineseMorphExtractor()

    def forward(self, text: str) -> torch.Tensor:
        """
        计算语义漂移系数S(t)

        Args:
            text: 输入的中文文本

        Returns:
            S_t: [seq_len] 语义漂移系数，取值范围[0,1]
        """
        # 从LLM提取语义状态向量h(t)
        hidden_states, token_num, tokenizer, inputs, attentions = extract_hidden_states(
            text=text,
            model_name=self.model_name,
            middle_layer_idx=self.middle_layer_idx,
            device=self.device
        )

        # 解码tokens
        tokens = []
        for i in range(len(inputs['input_ids'][0])):
            token_text = tokenizer.decode([inputs['input_ids'][0][i]])
            tokens.append(token_text)

        seq_len, d_model = hidden_states.shape

        # 【改进1】L2归一化：消除首Token范数异常影响
        if self.normalize_hidden:
            hidden_states = F.normalize(hidden_states, p=2, dim=1)  # 每个向量范数归一化为1

        # 【改进2】全局锚点：使用整个序列的平均表示（更符合"全局主题"定义）
        if self.use_global_anchor:
            # 全局句子级锚点：整个序列的加权平均
            h_global_fixed = self._compute_sentence_embedding(hidden_states)
        else:
            h_global_fixed = None

        # 初始化输出张量
        S_t = torch.zeros(seq_len, device=hidden_states.device, dtype=hidden_states.dtype)

        for t in range(seq_len):
            # 当前时间步的语义向量
            h_current = hidden_states[t]  # [d_model]

            # 计算全局语义锚点h_global(t)
            if self.use_global_anchor:
                h_global = h_global_fixed  # 使用固定的全局主题
            else:
                h_global = self._compute_global_anchor(hidden_states, t)  # 使用动态锚点

            # 计算归一化距离dist
            dist = self._compute_normalized_distance(h_current, h_global)

            # 计算dist_max(t)
            dist_max = self._compute_dist_max(hidden_states, t)

            # 距离比例
            dist_ratio = dist / (dist_max + 1e-8)

            # 计算M(t)
            m_t = self._compute_m_t(hidden_states, tokens, t, tokenizer)

            # 计算C(t)
            c_t = self._compute_c_t(hidden_states, tokens, t, m_t)

            # 计算D(t)
            d_t = self._compute_d_t(hidden_states, tokens, t, m_t)

            # 局部稳定性修正项
            stability_term = 1.0 - self.omega * c_t + self.nu * d_t

            # 中文形态修正因子ξ(M(t))
            xi = 1.0 - self.mu * m_t

            # 计算S(t)
            s_t_value = dist_ratio * stability_term * xi

            # 确保范围在[0, 1]
            S_t[t] = torch.clamp(s_t_value, 0.0, 1.0)

        return S_t

    def _compute_global_anchor(self, h_sequence: torch.Tensor, t: int) -> torch.Tensor:
        """
        计算全局语义锚点h_global(t) = Σ_{s=1}^t α(s,t) · h(s)
        
        公式(7-1): α(s,t) = softmax(h(s)^T·h(t) / (√d·exp(λ·(t-s))))

        Args:
            h_sequence: [seq_len, d_model] 序列语义向量
            t: 当前时间步

        Returns:
            h_global: [d_model] 全局锚点
        """
        # 过去的时间步h(s) for s=1 to t
        h_past = h_sequence[:t+1]  # [t+1, d_model]
        h_current = h_sequence[t]  # [d_model]
        d_model = h_sequence.shape[1]

        # 计算相似度: h(s)^T · h(t)
        similarity = torch.matmul(h_past, h_current)  # [t+1]

        # 时间衰减因子: exp(λ·(t-s))
        time_diff = torch.arange(t+1, dtype=torch.float32, device=h_sequence.device)
        time_decay = torch.exp(self.lambda_decay * (t - time_diff))  # exp(λ·(t-s))，近期token权重更大

        # 注意力logits: h(s)^T·h(t) / (√d·exp(λ·(t-s)))
        # 添加√d归一化项
        sqrt_d = torch.sqrt(torch.tensor(d_model, dtype=torch.float32, device=h_sequence.device))
        attention_logits = similarity / (sqrt_d * time_decay + 1e-8)

        # Softmax归一化
        alpha = F.softmax(attention_logits, dim=0)  # [t+1]

        # 加权求和
        h_global = torch.sum(h_past * alpha.unsqueeze(1), dim=0)  # [d_model]

        return h_global

    def _compute_sentence_embedding(self, h_sequence: torch.Tensor) -> torch.Tensor:
        """
        计算全局句子级embedding（非因果）
        
        使用整个序列的加权平均作为"全局主题"锚点
        这样"江河湖海"都会与同一个固定主题比较

        Args:
            h_sequence: [seq_len, d_model] 完整序列语义向量

        Returns:
            h_global_fixed: [d_model] 全局句子embedding
        """
        seq_len = h_sequence.shape[0]
        
        # 方案1：简单平均（最稳定）
        # h_global = torch.mean(h_sequence, dim=0)
        
        # 方案2：使用最后几个Token的平均（通常包含最完整信息）
        # last_k = min(3, seq_len)
        # h_global = torch.mean(h_sequence[-last_k:], dim=0)
        
        # 方案3：使用注意力加权（所有Token对所有Token的平均相似度）
        # 计算所有Token间的平均相似度作为权重
        similarity_matrix = torch.matmul(h_sequence, h_sequence.T)  # [seq_len, seq_len]
        avg_similarity = torch.mean(similarity_matrix, dim=1)  # [seq_len] 每个Token的平均相似度
        weights = F.softmax(avg_similarity, dim=0)  # 归一化为权重
        h_global = torch.sum(h_sequence * weights.unsqueeze(1), dim=0)  # 加权求和
        
        return h_global

    def _compute_normalized_distance(self, h_current: torch.Tensor, h_global: torch.Tensor) -> torch.Tensor:
        """
        计算欧氏距离
        
        如果已L2归一化（normalize_hidden=True）：
            - 所有向量范数=1，在单位球面上
            - 直接返回欧氏距离，范围∈[0, 2]
        
        如果未归一化（normalize_hidden=False）：
            - 除以max(||h(t)||, ||h_global||)进行范数归一化
            - 归一化后距离范围∈[0, √2]

        Args:
            h_current: [d_model] 当前语义向量
            h_global: [d_model] 全局锚点

        Returns:
            dist: 标量，欧氏距离
        """
        # 欧氏距离 ||h(t) - h_global(t)||_2
        euclidean_dist = torch.norm(h_current - h_global, p=2)

        # 如果已L2归一化，直接返回欧氏距离
        if self.normalize_hidden:
            return euclidean_dist
        
        # 未归一化时，除以max(norm)进行归一化
        norm_current = torch.norm(h_current, p=2)
        norm_global = torch.norm(h_global, p=2)
        denominator = torch.max(norm_current, norm_global)
        
        # 避免除零
        if denominator < 1e-8:
            return torch.tensor(0.0, device=h_current.device, dtype=h_current.dtype)

        return euclidean_dist / denominator

    def _compute_dist_max(self, h_sequence: torch.Tensor, t: int) -> torch.Tensor:
        """
        计算dist_max(t): 当前上下文窗口内的最大归一化距离
        
        由于归一化距离的理论上界是√2，这里计算实际观测到的最大距离

        Args:
            h_sequence: [seq_len, d_model] 序列语义向量
            t: 当前时间步

        Returns:
            dist_max: 标量，当前上下文的最大归一化距离
        """
        # 计算当前位置的全局锚点（已在forward中计算，这里重新计算以保持独立性）
        h_global = self._compute_global_anchor(h_sequence, t)
        
        # 计算从位置0到t的所有归一化距离
        max_dist = torch.tensor(0.0, device=h_sequence.device, dtype=h_sequence.dtype)
        for s in range(t + 1):
            dist = self._compute_normalized_distance(h_sequence[s], h_global)
            max_dist = torch.max(max_dist, dist)
        
        # 如果max_dist过小，使用理论上界√2的一半作为默认值
        if max_dist < 1e-6:
            max_dist = torch.tensor(1.0, device=h_sequence.device, dtype=h_sequence.dtype)

        return max_dist

    def _compute_m_t(self, h_t: torch.Tensor, tokens: List[str], t: int, tokenizer) -> float:
        """计算形态-语义匹配度M(t)"""
        token_text = tokens[t]
        return compute_m_t_full(
            h_t=h_t[t],
            token_text=token_text,
            tokens=tokens,
            token_idx=t,
            hidden_states=h_t,
            model=None,  # 已经在外部提取了hidden_states，这里不需要model
            tokenizer=tokenizer,
            layer_idx=self.middle_layer_idx
        )

    def _compute_c_t(self, h_t: torch.Tensor, tokens: List[str], t: int, m_t: float) -> float:
        """计算聚类密度C(t)"""
        return compute_c_t(
            h_t=h_t[t],
            hidden_states=h_t,
            token_idx=t,
            k=3,
            theta=0.5,
            alpha=0.4,
            precomputed_m_t=m_t
        )

    def _compute_d_t(self, h_t: torch.Tensor, tokens: List[str], t: int, m_t: float) -> float:
        """计算平均欧氏距离D(t)"""
        return compute_d_t(
            h_t=h_t[t],
            hidden_states=h_t,
            token_idx=t,
            sentence_length=len(tokens),
            window_size=3,
            sim_threshold=0.5,
            precomputed_m_t=m_t
        )
