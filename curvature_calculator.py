import torch
import numpy as np
from scipy.spatial.distance import cosine
from typing import Optional

def compute_local_sectional_curvature(
    h_t: torch.Tensor,
    hidden_states: torch.Tensor,
    token_idx: int,
    sentence_length: int,
    window_size: int = 3,
    sim_threshold: float = 0.5,
    gram_threshold: float = 1e-4,
    alpha: float = 0.25,
    epsilon: float = 1e-8,
    precomputed_m_t: Optional[float] = None  # 仅新增：接收外部真实M(t)
) -> float:
    """
    计算中文LLM语义层的局部截面曲率K(t)
    新增precomputed_m_t参数：优先使用外部传入的真实形态-语义匹配度M(t)
    """
    # 1. 转换为numpy数组（方便计算）
    h_t_np = h_t.detach().cpu().numpy() if h_t.requires_grad else h_t.cpu().numpy()
    hidden_states_np = hidden_states.detach().cpu().numpy() if hidden_states.requires_grad else hidden_states.cpu().numpy()

    # 2. 构建局部上下文窗口（token_idx ± window_size）
    start_idx = max(0, token_idx - window_size)
    end_idx = min(sentence_length - 1, token_idx + window_size)
    local_window = hidden_states_np[start_idx:end_idx+1]

    # 3. 语义相似度筛选有效邻域向量
    # 3.1 计算余弦相似度
    similarities = []
    for vec in local_window:
        sim = 1 - cosine(h_t_np, vec)
        similarities.append(sim)
    similarities = np.array(similarities)

    # 3.2 筛选有效向量（相似度≥阈值）
    valid_mask = similarities >= sim_threshold
    valid_vectors = local_window[valid_mask]
    if len(valid_vectors) < 2:
        return 0.0  # 有效向量不足，曲率为0

    # 3.3 中文形态-语义匹配度M(t)（优先用外部传入的真实值）
    if precomputed_m_t is not None:
        M_t = precomputed_m_t  # 复用外部真实M(t)
    else:
        # 保留原有模拟逻辑（兼容旧调用）
        hidden_dim = h_t_np.shape[0]
        morph_vec = np.random.randn(hidden_dim)
        M_t = 1 - cosine(h_t_np, morph_vec)
    M_t = max(M_t, 0.0)  # 截断负相关值

    # 4. 格拉姆-施密特正交化（构建正交基）
    try:
        # 使用QR分解实现正交化（替代gram_schmidt）
        Q, R = np.linalg.qr(valid_vectors.T)
        ortho_basis = Q.T  # (n, d)
        # 移除零向量列
        ortho_basis = ortho_basis[~np.all(np.abs(ortho_basis) < 1e-10, axis=1)]
        if ortho_basis.shape[0] < 2:
            return 0.0
    except Exception as e:
        return 0.0

    # 5. 计算截面曲率（带形态修正）
    # 5.1 计算格拉姆矩阵行列式
    gram_matrix = np.dot(ortho_basis, ortho_basis.T)
    gram_det = np.linalg.det(gram_matrix)
    gram_det = max(gram_det, gram_threshold)  # 避免行列式过小

    # 5.2 计算曲率核心项
    curvature_core = (np.sum(similarities) / len(similarities)) / (gram_det + epsilon)

    # 5.3 形态-语义耦合修正
    curvature = curvature_core * (1 + alpha * M_t)

    # 6. 归一化到[-1, 1]
    curvature = np.clip(curvature, -1.0, 1.0)
    return float(curvature)

# 批量计算函数（适配新增参数）
def compute_curvature_batch(
    hidden_states: torch.Tensor,
    sentence_length: int,
    window_size: int = 3,
    sim_threshold: float = 0.5,
    precomputed_m_t_list: Optional[list] = None  # 批量传入M(t)
) -> list:
    """批量计算所有token的曲率"""
    curvatures = []
    for token_idx in range(sentence_length):
        h_t = hidden_states[token_idx]
        # 匹配当前token的M(t)
        m_t = precomputed_m_t_list[token_idx] if precomputed_m_t_list else None
        k_t = compute_local_sectional_curvature(
            h_t=h_t,
            hidden_states=hidden_states,
            token_idx=token_idx,
            sentence_length=sentence_length,
            window_size=window_size,
            sim_threshold=sim_threshold,
            precomputed_m_t=m_t
        )
        curvatures.append(k_t)
    return curvatures