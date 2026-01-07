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
                 epsilon: float = 1e-6):
        """
        Args:
            decay_lambda: 时间衰减系数 λ
            omega: 聚类密度权重 ω
            nu: 平均距离权重 ν
            mu: 形态修正系数 μ
            epsilon: 正则化项
        """
        self.decay_lambda = decay_lambda
        self.omega = omega
        self.nu = nu
        self.mu = mu
        self.epsilon = epsilon
        
        # 初始化全局锚点计算器
        self.anchor_calculator = GlobalSemanticAnchor(decay_lambda=decay_lambda)
    
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
        distmax = 2.0  # 归一化欧氏距离的理论最大值
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
                         c_t_list: np.ndarray,
                         d_t_list: np.ndarray,
                         m_t_list: np.ndarray) -> np.ndarray:
        """批量计算序列中所有token的语义漂移系数
        
        Args:
            hidden_states: [seq_len, hidden_dim] 语义状态向量序列
            c_t_list: [seq_len] 聚类密度序列
            d_t_list: [seq_len] 平均欧氏距离序列
            m_t_list: [seq_len] 形态-语义匹配度序列
        
        Returns:
            s_t_array: [seq_len] 每个token的语义漂移系数
        """
        seq_len = hidden_states.shape[0]
        s_t_array = np.zeros(seq_len)
        
        for t in range(seq_len):
            # 计算全局语义锚点 G(t)
            anchor = self.anchor_calculator.compute_anchor(hidden_states, t)
            
            # 提取当前token的语义向量和相关指标
            h_t = hidden_states[t]
            c_t = c_t_list[t]
            d_t = d_t_list[t]
            m_t = m_t_list[t]
            
            # 计算语义漂移系数
            s_t = self.compute_s_t(h_t, anchor, c_t, d_t, m_t)
            s_t_array[t] = s_t
        
        return s_t_array


def init_drift_calculator(decay_lambda: float = 0.05,
                          omega: float = 0.4,
                          nu: float = 0.3,
                          mu: float = 0.25) -> SemanticDriftCalculator:
    """初始化语义漂移系数计算器（便于main.py调用）
    
    Args:
        decay_lambda: 时间衰减系数（默认0.05，适配中文连贯性）
        omega: 聚类密度权重（默认0.4）
        nu: 平均距离权重（默认0.3）
        mu: 形态修正系数（默认0.25）
    
    Returns:
        drift_calculator: 语义漂移系数计算器实例
    """
    return SemanticDriftCalculator(
        decay_lambda=decay_lambda,
        omega=omega,
        nu=nu,
        mu=mu
    )
