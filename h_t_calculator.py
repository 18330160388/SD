"""
中文大语言模型语义层多义性熵 H(t) 计算器

基于文档 3-3 定义：
H(t) = (1/log|S_t|) * [-∑_{s∈S_t} p(s|t)·log(p(s|t) + ε)] * ζ(t)

核心组件：
1. 核心义项集合 S_t：基于中文多义词词典
2. 义项激活概率 p(s|t)：融合语义、上下文、形态、句法特征
3. 归一化因子：1/log|S_t| 确保跨token可比
4. 语境修正因子 ζ(t)：适配中文搭配特性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class PolysemyDictionary:
    """中文多义词词典管理器
    
    基于《现代汉语词典》常见多义词，记录核心义项数量
    实际应用中可扩展为完整词典资源
    """
    
    def __init__(self):
        # 常见中文多义词及其核心义项数量（|S_t| ∈ [2, 15]）
        self.polysemy_dict = {
            # 高频多义动词
            "打": 10,  # 击打/打球/打电话/打印/打扫/打算/打折/打工/打架/打针
            "行": 5,   # 行走/银行/可行/品行/行业
            "过": 8,   # 经过/过去/过错/过分/过程/过滤/过节/过度
            "看": 7,   # 观看/看待/看病/看管/看望/看书/看法
            "上": 6,   # 上升/上学/上班/上网/上当/上级
            "下": 6,   # 下降/下课/下班/下雨/下手/下级
            "出": 8,   # 出现/出去/出差/出版/出口/出色/出生/出发
            "来": 7,   # 来到/未来/来自/来源/来往/来得及/原来
            "开": 9,   # 打开/开始/开会/开车/开花/开心/开发/开放/开水
            "发": 8,   # 发现/发展/发生/发送/发财/发烧/发言/发达
            
            # 高频多义名词
            "手": 4,   # 手部/手段/手艺/手下
            "头": 5,   # 头部/开头/头目/头等/头绪
            "面": 6,   # 脸面/表面/方面/面子/面粉/面积
            "点": 7,   # 地点/时点/点子/点心/特点/点火/点头
            "道": 5,   # 道路/说道/知道/道理/道德
            
            # 高频多义形容词
            "大": 5,   # 体积大/年龄大/大致/大人/大事
            "小": 5,   # 体积小/年龄小/小心/小人/小事
            "高": 6,   # 位置高/身高/高兴/高尚/高级/高超
            "长": 4,   # 长度/成长/长处/长官
            "深": 5,   # 深度/深刻/深夜/深入/深奥
            
            # 其他常见多义词
            "生": 6,   # 生命/生产/生长/学生/生疏/生气
            "老": 5,   # 年老/老师/老板/老实/老化
            "新": 4,   # 新的/新闻/新手/新颖
            "重": 5,   # 重量/重要/重复/重视/重新
            "轻": 4,   # 重量轻/轻松/轻视/轻微
        }
        
        # 默认义项数量（非多义词或未登录词）
        self.default_sense_count = 1
        
        # 义项数量范围约束
        self.min_senses = 2
        self.max_senses = 15
    
    def get_sense_count(self, token: str) -> int:
        """获取token的核心义项数量"""
        return self.polysemy_dict.get(token, self.default_sense_count)
    
    def is_polysemous(self, token: str) -> bool:
        """判断是否为多义词"""
        return self.get_sense_count(token) >= self.min_senses


class SenseActivationModel(nn.Module):
    """义项激活概率模型
    
    p(s|t) = softmax(MLP(Concat(h(t), c(t), m(t), syn(t))))
    
    融合四大特征：
    - h(t): 当前token语义向量
    - c(t): 上下文特征（窗口t-3~t+3的注意力加权）
    - m(t): 形态-语义特征（来自m_t_calculator）
    - syn(t): 句法搭配特征（简化为位置和邻域特征）
    """
    
    def __init__(self, hidden_dim: int = 896, morph_dim: int = 224, 
                 context_window: int = 3, max_senses: int = 15):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.morph_dim = morph_dim
        self.context_window = context_window
        self.max_senses = max_senses
        
        # 上下文特征提取（注意力加权机制）
        self.context_attention = nn.Linear(hidden_dim, 1)
        
        # 句法特征提取（简化为位置嵌入+邻域语义）
        self.syntax_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # 左右邻居语义
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # 融合MLP：h(d) + c(d) + m(224) + syn(64) → sense_dim
        input_dim = hidden_dim + hidden_dim + morph_dim + 64
        self.sense_mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, max_senses)  # 输出最大义项数的logits
        )
    
    def extract_context_features(self, hidden_states: torch.Tensor, 
                                  token_idx: int, 
                                  attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """提取上下文特征 c(t)
        
        Args:
            hidden_states: [seq_len, hidden_dim]
            token_idx: 当前token索引
            attention_weights: [seq_len, seq_len] 注意力权重矩阵
        
        Returns:
            context_vector: [hidden_dim] 上下文加权向量
        """
        seq_len = hidden_states.size(0)
        
        # 定义上下文窗口 [t-3, t+3]
        start_idx = max(0, token_idx - self.context_window)
        end_idx = min(seq_len, token_idx + self.context_window + 1)
        
        # 提取窗口内的hidden states
        context_hiddens = hidden_states[start_idx:end_idx]  # [window_size, hidden_dim]
        
        # 检查注意力权重是否可用且为二维矩阵 [seq_len, seq_len]
        use_external_attention = (
            attention_weights is not None and 
            attention_weights.dim() == 2 and 
            attention_weights.size(0) == seq_len and
            attention_weights.size(1) == seq_len
        )
        
        if use_external_attention:
            # 使用当前token对窗口内token的注意力作为权重
            attn_weights = attention_weights[token_idx, start_idx:end_idx]  # [window_size]
            attn_weights = F.softmax(attn_weights, dim=0).unsqueeze(-1)  # [window_size, 1]
            context_vector = (context_hiddens * attn_weights).sum(dim=0)  # [hidden_dim]
        else:
            # 无有效注意力权重时，使用可学习的注意力
            attn_scores = self.context_attention(context_hiddens).squeeze(-1)  # [window_size]
            attn_weights = F.softmax(attn_scores, dim=0).unsqueeze(-1)  # [window_size, 1]
            context_vector = (context_hiddens * attn_weights).sum(dim=0)  # [hidden_dim]
        
        return context_vector
    
    def extract_syntax_features(self, hidden_states: torch.Tensor, 
                               token_idx: int) -> torch.Tensor:
        """提取句法搭配特征 syn(t)
        
        简化实现：提取左右邻居的语义向量作为句法搭配信号
        
        Args:
            hidden_states: [seq_len, hidden_dim]
            token_idx: 当前token索引
        
        Returns:
            syntax_vector: [64] 句法特征向量
        """
        seq_len = hidden_states.size(0)
        
        # 提取左右邻居
        left_neighbor = hidden_states[token_idx - 1] if token_idx > 0 else hidden_states[token_idx]
        right_neighbor = hidden_states[token_idx + 1] if token_idx < seq_len - 1 else hidden_states[token_idx]
        
        # 拼接左右邻居作为句法搭配特征
        syntax_input = torch.cat([left_neighbor, right_neighbor], dim=0)  # [2 * hidden_dim]
        syntax_vector = self.syntax_proj(syntax_input)  # [64]
        
        return syntax_vector
    
    def forward(self, hidden_state: torch.Tensor, 
                context_feature: torch.Tensor,
                morph_feature: torch.Tensor,
                syntax_feature: torch.Tensor,
                num_senses: int) -> torch.Tensor:
        """计算义项激活概率分布
        
        Args:
            hidden_state: [hidden_dim] 当前token语义向量 h(t)
            context_feature: [hidden_dim] 上下文特征 c(t)
            morph_feature: [morph_dim] 形态特征 m(t)
            syntax_feature: [64] 句法特征 syn(t)
            num_senses: 当前token的核心义项数量 |S_t|
        
        Returns:
            sense_probs: [num_senses] 归一化的义项激活概率
        """
        # 拼接所有特征
        fused_features = torch.cat([
            hidden_state,
            context_feature,
            morph_feature,
            syntax_feature
        ], dim=0)  # [hidden_dim + hidden_dim + morph_dim + 64]
        
        # MLP映射到义项空间
        sense_logits = self.sense_mlp(fused_features)  # [max_senses]
        
        # 只取前num_senses个logits并归一化
        sense_logits = sense_logits[:num_senses]  # [num_senses]
        sense_probs = F.softmax(sense_logits, dim=0)  # [num_senses]
        
        return sense_probs


class PolysemyEntropyCalculator:
    """中文LLM语义层多义性熵 H(t) 计算器
    
    完整实现文档3-3定义的多义性熵计算管线
    """
    
    def __init__(self, hidden_dim: int = 896, morph_dim: int = 224,
                 epsilon: float = 1e-8, gamma: float = 0.08):
        """
        Args:
            hidden_dim: 语义向量维度
            morph_dim: 形态特征维度
            epsilon: 正则化项，避免log(0)
            gamma: 语境修正因子权重系数
        """
        self.hidden_dim = hidden_dim
        self.morph_dim = morph_dim
        self.epsilon = epsilon
        self.gamma = gamma
        
        # 初始化多义词词典
        self.polysemy_dict = PolysemyDictionary()
        
        # 初始化义项激活模型
        self.sense_model = SenseActivationModel(
            hidden_dim=hidden_dim,
            morph_dim=morph_dim
        )
    
    def compute_collocation_strength(self, tokens: List[str], 
                                    token_idx: int) -> float:
        """计算固定搭配强度 colloc(t)
        
        简化实现：基于常见搭配模式的启发式规则
        
        Args:
            tokens: 完整token序列
            token_idx: 当前token索引
        
        Returns:
            colloc_strength: 搭配强度 ∈ [0, 1]
        """
        # 常见强搭配模式（可扩展为搭配词典）
        strong_collocations = {
            # "打"的常见搭配
            ("打", "电话"): 1.0,
            ("打", "球"): 0.9,
            ("打", "针"): 0.85,
            ("打", "工"): 0.8,
            ("打", "补"): 0.85,  # "打补丁"的一部分
            ("打", "架"): 0.9,
            ("打", "印"): 0.85,
            ("打", "扫"): 0.8,
            # "行"的搭配
            ("行", "业"): 0.9,
            ("银", "行"): 1.0,
            # "看"的搭配
            ("看", "书"): 0.8,
            ("看", "病"): 0.85,
            # "开"的搭配
            ("开", "会"): 0.9,
            ("开", "车"): 0.85,
        }
        
        # 支持跨token的多字搭配（如"打补丁"、"打电话"）
        multi_token_collocations = {
            ("打", "补", "丁"): 0.95,  # "打补丁"
            ("打", "电", "话"): 1.0,   # "打电话"
            ("打", "电", "脑"): 0.7,   # "打电脑"（弱搭配）
        }
        
        # 检查左右邻居是否构成强搭配
        current_token = tokens[token_idx]
        max_strength = 0.2  # 默认弱搭配
        
        # 检查左侧单字搭配
        if token_idx > 0:
            left_pair = (tokens[token_idx - 1], current_token)
            max_strength = max(max_strength, strong_collocations.get(left_pair, 0.2))
        
        # 检查右侧单字搭配
        if token_idx < len(tokens) - 1:
            right_pair = (current_token, tokens[token_idx + 1])
            max_strength = max(max_strength, strong_collocations.get(right_pair, 0.2))
        
        # 检查三字搭配（当前token + 右侧两个token）
        if token_idx < len(tokens) - 2:
            tri_pattern = (current_token, tokens[token_idx + 1], tokens[token_idx + 2])
            max_strength = max(max_strength, multi_token_collocations.get(tri_pattern, 0.2))
        
        # 检查三字搭配（左侧一个 + 当前token + 右侧一个）
        if token_idx > 0 and token_idx < len(tokens) - 1:
            tri_pattern = (tokens[token_idx - 1], current_token, tokens[token_idx + 1])
            max_strength = max(max_strength, multi_token_collocations.get(tri_pattern, 0.2))
        
        return max_strength
    
    def compute_context_correction_factor(self, raw_entropy: float,
                                         global_mean_entropy: float,
                                         colloc_strength: float) -> float:
        """计算中文语境修正因子 ζ(t)
        
        ζ(t) = 1 + γ·sign(H0(t) - H̄)·colloc(t)
        
        Args:
            raw_entropy: 未修正的归一化熵 H0(t)
            global_mean_entropy: 全局平均熵 H̄
            colloc_strength: 固定搭配强度 colloc(t)
        
        Returns:
            correction_factor: 修正因子 ζ(t) ∈ [0.9, 1.1]
        """
        sign_term = 1.0 if raw_entropy > global_mean_entropy else -1.0
        correction = 1.0 + self.gamma * sign_term * colloc_strength
        
        # 限制修正范围 [0.9, 1.1]
        correction = max(0.9, min(1.1, correction))
        
        return correction
    
    def compute_entropy_for_token(self, token: str,
                                  sense_probs: torch.Tensor,
                                  num_senses: int,
                                  correction_factor: float = 1.0) -> float:
        """计算单个token的多义性熵 H(t)
        
        H(t) = (1/log|S_t|) * [-∑_{s∈S_t} p(s|t)·log(p(s|t) + ε)] * ζ(t)
        
        Args:
            token: 当前token字符串
            sense_probs: [num_senses] 义项激活概率分布
            num_senses: 核心义项数量 |S_t|
            correction_factor: 语境修正因子 ζ(t)
        
        Returns:
            entropy: 归一化多义性熵 H(t) ∈ [0, 1]
        """
        # 非多义词返回0
        if num_senses < 2:
            return 0.0
        
        # 计算香农熵：-∑ p(s|t)·log(p(s|t) + ε)
        log_probs = torch.log(sense_probs + self.epsilon)
        shannon_entropy = -(sense_probs * log_probs).sum().item()
        
        # 归一化：除以 log|S_t|
        normalization_factor = np.log(num_senses)
        normalized_entropy = shannon_entropy / normalization_factor
        
        # 应用语境修正因子
        final_entropy = normalized_entropy * correction_factor
        
        # 确保范围 [0, 1]
        final_entropy = max(0.0, min(1.0, final_entropy))
        
        return final_entropy
    
    def compute_batch_entropy(self, tokens: List[str],
                             hidden_states: torch.Tensor,
                             morph_features: torch.Tensor,
                             attention_weights: Optional[torch.Tensor] = None) -> np.ndarray:
        """批量计算序列中所有token的多义性熵
        
        Args:
            tokens: token序列 (长度T)
            hidden_states: [T, hidden_dim] 语义状态向量
            morph_features: [T, morph_dim] 形态特征向量
            attention_weights: [T, T] 注意力权重矩阵（可选）
        
        Returns:
            entropies: [T] 每个token的多义性熵
        """
        seq_len = len(tokens)
        entropies = np.zeros(seq_len)
        raw_entropies = []  # 用于计算全局平均
        
        # 第一遍：计算原始熵（不含修正因子）
        for t in range(seq_len):
            token = tokens[t]
            num_senses = self.polysemy_dict.get_sense_count(token)
            
            if not self.polysemy_dict.is_polysemous(token):
                raw_entropies.append(0.0)
                continue
            
            # 提取特征
            h_t = hidden_states[t]  # [hidden_dim]
            c_t = self.sense_model.extract_context_features(
                hidden_states, t, attention_weights
            )  # [hidden_dim]
            
            # 将 numpy array 转换为 Tensor（如果需要）
            m_t_raw = morph_features[t]  # [morph_dim] 可能是 numpy.ndarray
            if isinstance(m_t_raw, np.ndarray):
                m_t = torch.from_numpy(m_t_raw).float()
            else:
                m_t = m_t_raw
            
            syn_t = self.sense_model.extract_syntax_features(hidden_states, t)  # [64]
            
            # 计算义项激活概率
            with torch.no_grad():
                sense_probs = self.sense_model(h_t, c_t, m_t, syn_t, num_senses)
            
            # 计算原始熵（不含修正）
            log_probs = torch.log(sense_probs + self.epsilon)
            shannon_entropy = -(sense_probs * log_probs).sum().item()
            normalized_entropy = shannon_entropy / np.log(num_senses)
            
            raw_entropies.append(normalized_entropy)
        
        # 计算全局平均熵
        global_mean_entropy = np.mean([e for e in raw_entropies if e > 0]) if any(raw_entropies) else 0.5
        
        # 第二遍：应用语境修正因子
        for t in range(seq_len):
            token = tokens[t]
            num_senses = self.polysemy_dict.get_sense_count(token)
            
            if not self.polysemy_dict.is_polysemous(token):
                entropies[t] = 0.0
                continue
            
            # 计算搭配强度
            colloc_strength = self.compute_collocation_strength(tokens, t)
            
            # 计算修正因子
            correction_factor = self.compute_context_correction_factor(
                raw_entropies[t], global_mean_entropy, colloc_strength
            )
            
            # 应用修正
            entropies[t] = raw_entropies[t] * correction_factor
            entropies[t] = max(0.0, min(1.0, entropies[t]))
        
        return entropies


def init_entropy_calculator(hidden_dim: int = 896, 
                           morph_dim: int = 224) -> PolysemyEntropyCalculator:
    """初始化多义性熵计算器（便于main.py调用）"""
    return PolysemyEntropyCalculator(
        hidden_dim=hidden_dim,
        morph_dim=morph_dim
    )
