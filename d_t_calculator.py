import torch
import numpy as np
from typing import Optional

def compute_d_t(
    h_t: torch.Tensor,
    hidden_states: torch.Tensor,
    token_idx: int,
    sentence_length: int,
    window_size: int = 3,
    sim_threshold: float = 0.5,
    epsilon: float = 1e-6,
    precomputed_m_t: Optional[float] = None,
    return_diag: bool = False,
) -> float:
    """
    计算局部语义分布距离 D(t)（严格按文档定义）
    
    D(t) = 1/N_pair * Σ_{i<j∈S_eff} ||h_i - h_j||² / max(||h_i||², ||h_j||²) * β(M(t))
    
    物理意义：局部上下文窗口内有效向量的平均归一化欧氏距离，反映语义分散程度
    
    Args:
        h_t: 当前token的隐藏状态向量，shape=(hidden_dim,)
        hidden_states: 整句话的隐藏状态，shape=(token_num, hidden_dim)
        token_idx: 当前token的索引
        sentence_length: 句子的token总数
        window_size: 局部上下文窗口大小 k（窗口范围[t-k, t+k]）
        sim_threshold: 语义相关性阈值 θ（余弦相似度）
        epsilon: 正则化项（避免除零）
        precomputed_m_t: 预计算的形态-语义匹配度 M(t)
    
    Returns:
        D_t: 语义分布距离，标量值 ∈ [0, 2]
    """
    # 1. 转换为numpy数组
    h_t_np = h_t.detach().cpu().numpy() if h_t.requires_grad else h_t.cpu().numpy()
    hidden_states_np = hidden_states.detach().cpu().numpy() if hidden_states.requires_grad else hidden_states.cpu().numpy()
    
    # 2. 构建局部上下文窗口 [t-k, t+k]
    start_idx = max(0, token_idx - window_size)
    end_idx = min(sentence_length - 1, token_idx + window_size)
    local_window = hidden_states_np[start_idx:end_idx+1]
    
    # 3. 有效向量筛选：语义相关性（与 h_t 余弦相似度 ≥ θ）
    h_t_norm = h_t_np / (np.linalg.norm(h_t_np) + epsilon)
    S_eff = []
    S_eff_indices = []
    
    for idx, h_s in enumerate(local_window):
        global_idx = start_idx + idx
        # 排除自身
        if global_idx == token_idx:
            continue
        
        # 语义相关性筛选
        h_s_norm = h_s / (np.linalg.norm(h_s) + epsilon)
        cos_sim = np.dot(h_t_norm, h_s_norm)
        
        if cos_sim >= sim_threshold:
            S_eff.append(h_s)
            S_eff_indices.append(global_idx)
    
    # 记录原始有效集大小
    seff_size = len(S_eff)

    fallback_used = False

    # If there are no effective neighbors at all (Seff_size == 0),
    # return 0.0 directly to avoid fallback producing large noisy values.
    if seff_size == 0:
        D_t = 0.0
        if return_diag:
            diag = {
                'Seff_size': seff_size,
                'fallback_used': False,
                'M_t': float(precomputed_m_t) if precomputed_m_t is not None else None,
                'D_t': float(D_t)
            }
            return float(D_t), diag
        return float(D_t)

    # 如果有效向量数 < 2，使用fallback策略：
    # - 尝试使用局部窗口（排除自身）作为后备集合，计算 h_t 与每个后备向量的归一化距离平均值
    # - 仅在窗口为空时才返回0.0（极短句或仅1个token情况）
    if len(S_eff) < 2:
        fallback_used = True
        # 后备集合：local_window中排除自身
        fallback_candidates = []
        for idx, h_s in enumerate(local_window):
            global_idx = start_idx + idx
            if global_idx == token_idx:
                continue
            fallback_candidates.append(h_s)

        if len(fallback_candidates) == 0:
            D_t = 0.0
            if return_diag:
                diag = {
                    'Seff_size': seff_size,
                    'fallback_used': True,
                    'M_t': float(precomputed_m_t) if precomputed_m_t is not None else None,
                    'D_t': float(D_t)
                }
                return float(D_t), diag
            return float(D_t)

        # 计算 h_t 与每个后备向量的归一化欧氏距离（单边平均）
        fallback_arr = np.array(fallback_candidates)
        n_fb = fallback_arr.shape[0]
        total_fb_dist = 0.0
        h_t_vec = h_t_np
        h_t_norm = np.linalg.norm(h_t_vec)
        for h_s in fallback_arr:
            euclidean_dist = np.linalg.norm(h_t_vec - h_s)
            norm_i = h_t_norm
            norm_j = np.linalg.norm(h_s)
            norm_factor = max(norm_i, norm_j) + epsilon
            normalized_dist = euclidean_dist / norm_factor
            total_fb_dist += normalized_dist

        avg_distance = total_fb_dist / float(n_fb)
        # 已使用 fallback，直接应用形态修正并返回
        if fallback_used:
            if precomputed_m_t is not None:
                M_t = np.clip(precomputed_m_t, 0.0, 1.0)
                beta_M = 1.0 - 0.3 * M_t
            else:
                beta_M = 1.0

            D_t = (avg_distance * beta_M)
            if return_diag:
                diag = {
                    'Seff_size': seff_size,
                    'fallback_used': True,
                    'M_t': float(precomputed_m_t) if precomputed_m_t is not None else None,
                    'D_t': float(D_t)
                }
                return float(D_t), diag
            return float(D_t)
    
    S_eff = np.array(S_eff)
    
    # 4. 计算两两配对的归一化欧氏距离（使用 L2 距离 / L2 范数，按文档）
    n_eff = len(S_eff)
    N_pair = n_eff * (n_eff - 1) // 2  # 组合数 C(n,2)
    
    total_distance = 0.0
    for i in range(n_eff):
        for j in range(i + 1, n_eff):
            h_i = S_eff[i]
            h_j = S_eff[j]
            # L2 距离（非平方）
            euclidean_dist = np.linalg.norm(h_i - h_j)

            # 归一化因子：max(||h_i||, ||h_j||)
            norm_i = np.linalg.norm(h_i)
            norm_j = np.linalg.norm(h_j)
            norm_factor = max(norm_i, norm_j) + epsilon

            # 归一化距离（文档：||h_i-h_j|| / max(||h_i||,||h_j||)）
            normalized_dist = euclidean_dist / norm_factor
            total_distance += normalized_dist
    
    # 5. 平均距离
    avg_distance = total_distance / N_pair
    
    # 6. 形态-语义匹配度修正因子 β(M(t))
    # 文档定义：β(M(t)) 为单调递减函数，M(t) 越高，修正越小
    # 实现：β(M(t)) = 1 - 0.3·M(t)，使得 β ∈ [0.7, 1.0]
    if precomputed_m_t is not None:
        M_t = np.clip(precomputed_m_t, 0.0, 1.0)
        beta_M = 1.0 - 0.3 * M_t  # M(t)高时，β小，D(t)被压缩
    else:
        beta_M = 1.0  # 默认无修正
    
    # 7. 最终 D(t)
    D_t = avg_distance * beta_M

    if return_diag:
        diag = {
            'Seff_size': seff_size,
            'fallback_used': bool(fallback_used),
            'M_t': float(precomputed_m_t) if precomputed_m_t is not None else None,
            'D_t': float(D_t)
        }
        return float(D_t), diag

    return float(D_t)


def compute_d_t_batch(
    hidden_states: torch.Tensor,
    window_size: int = 3,
    sim_threshold: float = 0.5,
    epsilon: float = 1e-6,
    precomputed_m_t_list: Optional[np.ndarray] = None,
    return_diagnostics: bool = False
) -> np.ndarray:
    """
    批量计算所有token的D(t)
    
    Args:
        hidden_states: 整句话的隐藏状态，shape=(token_num, hidden_dim)
        window_size: 局部上下文窗口大小
        sim_threshold: 语义相关性阈值
        epsilon: 正则化项
        precomputed_m_t_list: 预计算的 M(t) 数组，shape=(token_num,)
    
    Returns:
        D_t_batch: 所有token的D(t)数组，shape=(token_num,)
    """
    token_num = hidden_states.shape[0]
    D_t_list = []
    diagnostics = []
    
    for token_idx in range(token_num):
        h_t = hidden_states[token_idx]
        precomputed_m_t = precomputed_m_t_list[token_idx] if precomputed_m_t_list is not None else None
        
        result = compute_d_t(
            h_t=h_t,
            hidden_states=hidden_states,
            token_idx=token_idx,
            sentence_length=token_num,
            window_size=window_size,
            sim_threshold=sim_threshold,
            epsilon=epsilon,
            precomputed_m_t=precomputed_m_t
        , return_diag=return_diagnostics)

        if return_diagnostics:
            D_t, diag = result
            D_t_list.append(D_t)
            diagnostics.append(diag)
        else:
            D_t_list.append(result)
    
    D_arr = np.array(D_t_list)
    if return_diagnostics:
        return D_arr, diagnostics
    return D_arr


# 测试代码（单独运行验证）
if __name__ == "__main__":
    # 模拟隐藏状态（5个token，每个token维度10）
    test_hidden = torch.randn(5, 10)
    test_token_idx = 2
    
    # 计算单个token的D(t)
    d_t = compute_d_t(
        h_t=test_hidden[test_token_idx],
        hidden_states=test_hidden,
        token_idx=test_token_idx,
        sentence_length=5,
        window_size=3,
        sim_threshold=0.5,
        epsilon=1e-6
    )
    print(f"单个Token的D(t)值：{d_t:.6f}")
    
    # 批量计算所有token的D(t)
    d_t_batch = compute_d_t_batch(test_hidden, window_size=3, sim_threshold=0.5)
    print(f"\n所有Token的D(t)数组：")
    for idx, d in enumerate(d_t_batch):
        print(f"Token[{idx}] D(t) = {d:.6f}")