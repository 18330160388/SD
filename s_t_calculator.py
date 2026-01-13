"""
中文大语言模型语义层语义漂移系数 S(t) 计算器

基于文档 3-4 定义：
S(t) = [dist(h(t), G(t)) / (distmax(t) + ε)] × [1 - ω·C(t) + ν·D(t)] × ξ(M(t))

核心组件：
1. 全局语义锚点 G(t)：动态时间衰减自注意力加权
2. 归一化欧氏距离 dist(h, G)：局部-锚点偏离度
3. 局部稳定性修正：聚类密度 C(t) 和平均距离 D(t)
4. 中文形态修正因子：ξ(M(t)) = 1 - μ·M(t)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

# 导入依赖模块
from m_t_calculator import compute_m_t_full, ChineseMorphExtractor, MorphEmbedding
from c_t_calculator import compute_c_t
from d_t_calculator import compute_d_t
from h_t_calculator import init_entropy_calculator


class GlobalSemanticAnchor:
    """全局语义锚点计算器
    
    G(t) = Σ_{s=1}^{t} α(s,t)·h(s)
    α(s,t) = softmax(h(s)^T·h(t)/√d · exp(-λ·|t-s|))
    """
    
    def __init__(self, decay_lambda: float = 0.05):
        """
        Args:
            decay_lambda: 时间衰减系数 λ，强化近期token对主题的贡献
        """
        self.decay_lambda = decay_lambda
    
    def compute_attention_weights(self, 
                                  hidden_states: torch.Tensor,
                                  current_idx: int) -> torch.Tensor:
        """计算时间衰减自注意力权重 α(s,t)
        
        Args:
            hidden_states: [seq_len, hidden_dim] 所有token的语义向量
            current_idx: 当前token索引 t
        
        Returns:
            attention_weights: [current_idx+1] 归一化注意力权重
        """
        seq_len, hidden_dim = hidden_states.shape
        h_t = hidden_states[current_idx]  # [hidden_dim]
        
        # 只考虑当前token及之前的token（因果性）
        past_hidden = hidden_states[:current_idx + 1]  # [current_idx+1, hidden_dim]
        
        # 计算语义相似度：h(s)^T·h(t)/√d
        similarity = torch.matmul(past_hidden, h_t) / np.sqrt(hidden_dim)  # [current_idx+1]
        
        # 计算时间距离并应用指数衰减：exp(-λ·|t-s|)
        time_distances = torch.arange(current_idx + 1, dtype=torch.float32)
        time_decay = torch.exp(-self.decay_lambda * (current_idx - time_distances))
        
        # 融合语义相似度和时间衰减
        attention_logits = similarity * time_decay  # [current_idx+1]
        
        # Softmax归一化
        attention_weights = F.softmax(attention_logits, dim=0)
        
        return attention_weights
    
    def compute_anchor(self, 
                      hidden_states: torch.Tensor,
                      current_idx: int) -> torch.Tensor:
        """计算全局语义锚点 G(t)
        
        Args:
            hidden_states: [seq_len, hidden_dim] 所有token的语义向量
            current_idx: 当前token索引 t
        
        Returns:
            anchor: [hidden_dim] 全局语义锚点向量
        """
        # 计算注意力权重
        attention_weights = self.compute_attention_weights(hidden_states, current_idx)
        
        # 加权求和：G(t) = Σ α(s,t)·h(s)
        past_hidden = hidden_states[:current_idx + 1]  # [current_idx+1, hidden_dim]
        anchor = torch.sum(past_hidden * attention_weights.unsqueeze(1), dim=0)  # [hidden_dim]
        
        return anchor


class SemanticDriftCalculator:
    """语义漂移系数 S(t) 计算器
    
    完整实现文档 3-4 定义的语义漂移系数计算管线
    """
    
    def __init__(self, 
                 decay_lambda: float = 0.05,
                 omega: float = 0.4,
                 nu: float = 0.3,
                 mu: float = 0.25,
                 epsilon: float = 1e-6,
                 morph_extractor: Optional[ChineseMorphExtractor] = None,
                 morph_embedding: Optional[MorphEmbedding] = None,
                 h_t_calculator: Optional[object] = None,
                 model=None,  # 新增：LLM模型（用于Φ(m(t))）
                 tokenizer=None):  # 新增：分词器（用于Φ(m(t))）
        """
        Args:
            decay_lambda: 时间衰减系数 λ
            omega: 聚类密度权重 ω
            nu: 平均距离权重 ν
            mu: 形态修正系数 μ
            epsilon: 正则化项
            morph_extractor: 中文形态特征提取器（用于M(t)计算）
            morph_embedding: 形态嵌入模型（保留参数兼容性）
            h_t_calculator: 多义熵计算器（用于M(t)计算）
            model: LLM模型（用于提取字符embedding）
            tokenizer: 分词器（用于编码字符）
        """
        self.decay_lambda = decay_lambda
        self.omega = omega
        self.nu = nu
        self.mu = mu
        self.epsilon = epsilon
        
        # 初始化全局锚点计算器
        self.anchor_calculator = GlobalSemanticAnchor(decay_lambda=decay_lambda)
        
        # 初始化M(t)计算所需组件
        self.morph_extractor = morph_extractor if morph_extractor else ChineseMorphExtractor()
        self.morph_embedding = morph_embedding if morph_embedding else MorphEmbedding(morph_dim=254, hidden_dim=896)
        self.h_t_calculator = h_t_calculator if h_t_calculator else init_entropy_calculator()
        
        # 新增：保存model和tokenizer用于Φ(m(t))计算
        self.model = model
        self.tokenizer = tokenizer
    
    def compute_normalized_distance(self, 
                                    h_t: torch.Tensor,
                                    anchor: torch.Tensor) -> float:
        """计算归一化欧氏距离 dist(h(t), G(t))
        
        dist(h, G) = ||h - G||_2 / max(||h||_2, ||G||_2)
        
        Args:
            h_t: [hidden_dim] 当前token语义向量
            anchor: [hidden_dim] 全局语义锚点
        
        Returns:
            normalized_dist: 归一化距离 ∈ [0, 2]
        """
        # 计算欧氏距离
        euclidean_dist = torch.norm(h_t - anchor, p=2).item()
        
        # 计算归一化分母：max(||h||_2, ||G||_2)
        norm_h = torch.norm(h_t, p=2).item()
        norm_g = torch.norm(anchor, p=2).item()
        denominator = max(norm_h, norm_g) + self.epsilon
        
        # 归一化
        normalized_dist = euclidean_dist / denominator
        
        return normalized_dist
    
    def compute_morphology_correction_factor(self, m_t: float) -> float:
        """计算中文形态-语义匹配度修正因子 ξ(M(t))
        
        ξ(M(t)) = 1 - μ·M(t)
        
        形态匹配度高 → 语义稳定 → 漂移系数降低
        
        Args:
            m_t: 形态-语义匹配度 M(t) ∈ [0, 1]
        
        Returns:
            correction_factor: ξ(M(t)) ∈ [1-μ, 1]
        """
        return 1.0 - self.mu * m_t
    
    def compute_local_stability_correction(self, 
                                          c_t: float,
                                          d_t: float) -> float:
        """计算局部稳定性修正项
        
        修正项 = 1 - ω·C(t) + ν·D(t)
        
        - 聚类密度 C(t) 越高 → 局部越稳定 → 漂移系数降低
        - 平均距离 D(t) 越大 → 局部越分散 → 漂移系数升高
        
        Args:
            c_t: 聚类密度 C(t)
            d_t: 平均欧氏距离 D(t)
        
        Returns:
            correction: 局部稳定性修正系数
        """
        correction = 1.0 - self.omega * c_t + self.nu * d_t
        
        # 确保修正因子为正（避免极端情况）
        correction = max(0.1, correction)
        
        return correction
    
    def compute_s_t(self,
                   h_t: torch.Tensor,
                   anchor: torch.Tensor,
                   c_t: float,
                   d_t: float,
                   m_t: float) -> float:
        """计算单个token的语义漂移系数 S(t)
        
        S(t) = [dist(h(t), G(t)) / (distmax + ε)] × [1 - ω·C(t) + ν·D(t)] × ξ(M(t))
        
        Args:
            h_t: [hidden_dim] 当前token语义向量
            anchor: [hidden_dim] 全局语义锚点
            c_t: 聚类密度
            d_t: 平均欧氏距离
            m_t: 形态-语义匹配度
        
        Returns:
            s_t: 语义漂移系数 S(t) ∈ [0, 1]
        """
        # 1. 计算归一化锚点偏离度
        dist = self.compute_normalized_distance(h_t, anchor)
        
        # 自适应distmax：基于向量范数的理论上界
        # 对于L2归一化向量，最大距离为sqrt(2)≈1.414
        # 但实际LLM hidden states通常未归一化，使用更大的基准
        h_norm = torch.norm(h_t).item()
        anchor_norm = torch.norm(anchor).item()
        # 理论最大距离约为 ||h_t|| + ||G(t)||
        distmax = max(h_norm + anchor_norm, 1.0)  # 至少为1避免除0
        
        drift_base = dist / (distmax + self.epsilon)
        
        # 2. 计算局部稳定性修正
        stability_correction = self.compute_local_stability_correction(c_t, d_t)
        
        # 3. 计算中文形态修正因子
        morph_correction = self.compute_morphology_correction_factor(m_t)
        
        # 4. 计算最终漂移系数
        s_t = drift_base * stability_correction * morph_correction
        
        # 确保范围 [0, 1]
        s_t = max(0.0, min(1.0, s_t))
        
        return s_t
    
    def compute_s_t_batch(self,
                         hidden_states: torch.Tensor,
                         tokens: List[str],
                         attention_weights: Optional[torch.Tensor] = None) -> np.ndarray:
        """批量计算序列中所有token的语义漂移系数（完整实现）
        
        Args:
            hidden_states: [seq_len, hidden_dim] 语义状态向量序列
            tokens: [seq_len] token文本列表
            attention_weights: [seq_len, seq_len] 注意力权重矩阵（可选，用于M(t)计算）
        
        Returns:
            s_t_array: [seq_len] 每个token的语义漂移系数
        """
        seq_len = hidden_states.shape[0]
        s_t_array = np.zeros(seq_len)
        
        for t in range(seq_len):
            # 1. 计算全局语义锚点 G(t)
            anchor = self.anchor_calculator.compute_anchor(hidden_states, t)
            
            # 2. 提取当前token的语义向量和文本
            h_t = hidden_states[t]
            token_text = tokens[t]
            
            # 3. 计算形态-语义匹配度 M(t)（复用 m_t_calculator.py）
            m_t = compute_m_t_full(
                h_t=h_t,
                token_text=token_text,
                tokens=tokens,
                token_idx=t,
                hidden_states=hidden_states,
                model=self.model,  # 传入LLM模型
                tokenizer=self.tokenizer,  # 传入分词器
                attention_weights=attention_weights,
                beta=0.2
            )
            
            # 4. 计算聚类密度 C(t)（复用 c_t_calculator.py）
            c_t = compute_c_t(
                h_t=h_t,
                hidden_states=hidden_states,
                token_idx=t,
                k=3,  # 上下文窗口大小
                theta=0.5,  # 相似度阈值
                alpha=0.4,  # 形态修正权重
                precomputed_m_t=m_t  # 传入已计算的M(t)
            )
            
            # 5. 计算平均欧氏距离 D(t)（复用 d_t_calculator.py）
            d_t = compute_d_t(
                h_t=h_t,
                hidden_states=hidden_states,
                token_idx=t,
                sentence_length=seq_len,
                window_size=3,
                sim_threshold=0.5,
                precomputed_m_t=m_t  # 传入已计算的M(t)
            )
            
            # 6. 计算语义漂移系数 S(t)
            s_t = self.compute_s_t(h_t, anchor, c_t, d_t, m_t)
            s_t_array[t] = s_t
        
        return s_t_array


def init_drift_calculator(decay_lambda: float = 0.05,
                          omega: float = 0.4,
                          nu: float = 0.3,
                          mu: float = 0.25,
                          morph_extractor: Optional[ChineseMorphExtractor] = None,
                          morph_embedding: Optional[MorphEmbedding] = None,
                          h_t_calculator: Optional[object] = None,
                          model=None,  # 新增
                          tokenizer=None) -> SemanticDriftCalculator:  # 新增
    """初始化语义漂移系数计算器（便于main.py调用）
    
    Args:
        decay_lambda: 时间衰减系数（默认0.05，适配中文连贯性）
        omega: 聚类密度权重（默认0.4）
        nu: 平均距离权重（默认0.3）
        mu: 形态修正系数（默认0.25）
        morph_extractor: 形态特征提取器（可选，不提供则自动创建）
        morph_embedding: 形态嵌入模型（可选，不提供则自动创建）
        h_t_calculator: H(t)计算器（可选，不提供则自动创建）
        model: LLM模型（可选，用于Φ(m(t))）
        tokenizer: 分词器（可选，用于Φ(m(t))）
    
    Returns:
        drift_calculator: 语义漂移系数计算器实例
    """
    return SemanticDriftCalculator(
        decay_lambda=decay_lambda,
        omega=omega,
        nu=nu,
        mu=mu,
        morph_extractor=morph_extractor,
        morph_embedding=morph_embedding,
        h_t_calculator=h_t_calculator,
        model=model,
        tokenizer=tokenizer
    )
