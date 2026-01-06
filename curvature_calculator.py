import torch
import numpy as np
from scipy.spatial.distance import cosine

def compute_local_sectional_curvature(
    h_t: torch.Tensor,
    hidden_states: torch.Tensor,
    token_idx: int,
    sentence_length: int,
    window_size: int = 3,  # 局部窗口大小（左右各3个token）
    sim_threshold: float = 0.5,  # 语义相关性阈值
    gram_threshold: float = 1e-4,  # 线性无关性阈值
    alpha: float = 0.25,  # 形态修正因子权重
    epsilon: float = 1e-8  # 正则化项
) -> float:
    """
    计算第token_idx个token的局部截面曲率K(t)（严格遵循文档定义）
    
    Args:
        h_t: 当前token的语义向量h(t)，shape=(hidden_dim,)
        hidden_states: 整句的语义向量集合，shape=(token_num, hidden_dim)
        token_idx: 当前token的索引（0开始）
        sentence_length: 句子的token总数
        window_size: 局部上下文窗口大小（k值）
        sim_threshold: 余弦相似度阈值（筛选语义相关向量）
        gram_threshold: Gram行列式阈值（筛选线性无关向量）
        alpha: 形态修正因子的权重（0.2~0.3之间）
        epsilon: 避免分母为0的正则化项
    
    Returns:
        K_t: 局部截面曲率（float）
    """
    # ---------------------- 1. 局部上下文窗口筛选 ----------------------
    start_idx = max(0, token_idx - window_size)
    end_idx = min(sentence_length - 1, token_idx + window_size)
    # 提取窗口内所有向量（排除当前token自身）
    window_vectors = []
    for idx in range(start_idx, end_idx + 1):
        if idx != token_idx:
            window_vectors.append(hidden_states[idx].numpy())  # 转为numpy数组计算
    
    if not window_vectors:
        return 0.0  # 无有效上下文向量，返回平坦曲率
    
    # ---------------------- 2. 筛选核心上下文向量h*(t) ----------------------
    valid_vecs = []
    h_t_np = h_t.numpy()
    norm_h = np.linalg.norm(h_t_np, 2)  # h(t)的L2范数
    
    for vec in window_vectors:
        # 2.1 语义相关性：余弦相似度≥sim_threshold
        cos_sim = 1 - cosine(h_t_np, vec)
        if cos_sim < sim_threshold:
            continue
        
        # 2.2 线性无关性：Gram行列式≥gram_threshold
        norm_vec = np.linalg.norm(vec, 2)
        gram = (norm_h ** 2) * (norm_vec ** 2) - (np.dot(h_t_np, vec) ** 2)
        if gram < gram_threshold:
            continue
        
        # 2.3 计算评分（相似度×Gram行列式）
        score = cos_sim * gram
        valid_vecs.append((vec, score))
    
    if not valid_vecs:
        return 0.0  # 无有效h*(t)，返回平坦曲率
    
    # 选择评分最高的向量作为h*(t)
    h_star = max(valid_vecs, key=lambda x: x[1])[0]
    
    # ---------------------- 3. 计算中间指标（协方差、方差等） ----------------------
    def sample_cov(a: np.ndarray, b: np.ndarray) -> float:
        """样本协方差（无偏估计）"""
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        return np.sum((a - mean_a) * (b - mean_b)) / (len(a) - 1)
    
    # 3.1 协方差和方差
    cov_h_hstar = sample_cov(h_t_np, h_star)
    var_h = sample_cov(h_t_np, h_t_np)  # 方差=自协方差
    h_dot_hstar = np.dot(h_t_np, h_star)  # h(t)⊤h*(t)
    norm_h_sq = norm_h ** 2  # ||h(t)||²
    
    # 3.2 Gram行列式（重新计算确保准确）
    norm_hstar = np.linalg.norm(h_star, 2)
    gram = (norm_h_sq) * (norm_hstar ** 2) - (h_dot_hstar ** 2)
    
    # 3.3 中文形态-语义匹配度M(t)（简化实现，可替换为真实形态特征）
    # 真实场景：用Glyph-Aware工具提取形态向量，替换以下随机向量
    hidden_dim = h_t_np.shape[0]  # 获取实际的隐藏状态维度
    morph_vec = np.random.randn(hidden_dim)  # 示例：形态特征向量（部首+笔画）
    M_t = 1 - cosine(h_t_np, morph_vec)  # M(t)∈[0,1]
    
    # ---------------------- 4. 计算最终K(t) ----------------------
    # 分子
    numerator = cov_h_hstar * norm_h_sq - h_dot_hstar * var_h
    # 分母
    denominator = gram * var_h + epsilon
    # 原始曲率K0(t)
    K0 = numerator / denominator
    # 形态修正因子γ(M(t))
    gamma = 1 + alpha * np.sign(K0) * M_t
    # 最终曲率
    K_t = K0 * gamma
    
    return round(K_t, 6)  # 保留6位小数，便于后续分析

# 测试代码（单独运行时验证）
if __name__ == "__main__":
    # 模拟h(t)和hidden_states（实际使用时从llm_hidden_extractor导入）
    hidden_dim = 896  # 使用实际模型的隐藏维度
    test_hidden = torch.randn(5, hidden_dim)  # 模拟5个token的h(t)
    test_h_t = test_hidden[2]  # 第3个token（index=2）的h(t)
    
    K_t = compute_local_sectional_curvature(
        h_t=test_h_t,
        hidden_states=test_hidden,
        token_idx=2,
        sentence_length=5
    )
    print(f"测试token的局部截面曲率K(t)：{K_t}")