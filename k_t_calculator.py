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
    sim_threshold: float = 0.3,
    gram_threshold: float = 1e-4,
    alpha: float = 0.25,
    epsilon: float = 1e-8,
    precomputed_m_t: Optional[float] = None,
    unit_normalize: bool = True
) -> float:
    """
    计算局部截面曲率K(t) - 基于向量几何关系
    
    K(t) = [Cov(h(t),h*(t))·‖h(t)‖²₂ - (h(t)ᵀh*(t))·Cov(h(t),h(t))] / 
           [Gram(h(t),h*(t))·Var(h(t)) + ε] · γ(M(t))
    
    几何解释：
    - Cov(h(t),h*(t)): 两个向量间的几何协方差（夹角余弦）
    - Var(h(t)): 向量相对于局部上下文的离散程度（1-平均相似度）
    - Gram(h(t),h*(t)): 向量张成的平行四边形面积
    - γ(M(t)): 形态-语义修正因子
    
    注意：使用原始向量计算几何关系，确保数值稳定性
    """
    # 1. 转换为numpy数组
    h_t_np = h_t.detach().cpu().numpy() if h_t.requires_grad else h_t.cpu().numpy()
    hidden_states_np = hidden_states.detach().cpu().numpy() if hidden_states.requires_grad else hidden_states.cpu().numpy()

    # 2. 构建局部上下文窗口 [t-k, t+k]
    start_idx = max(0, token_idx - window_size)
    end_idx = min(sentence_length - 1, token_idx + window_size)
    local_window = hidden_states_np[start_idx:end_idx+1]
    
    # 3. 有效向量筛选：语义相关性 + 线性无关性（双筛选）
    # 用归一化向量做筛选（判断语义相似度）
    h_t_norm = h_t_np / (np.linalg.norm(h_t_np) + epsilon)
    S_eff = []
    S_eff_original = []  # 保存原始向量用于协方差计算
    S_eff_info = []  # 保存每个有效向量的详细信息（用于调试输出）
    
    for idx, h_s in enumerate(local_window):
        # 排除自身
        global_idx = start_idx + idx
        if global_idx == token_idx:
            continue
            
        # 3.1 语义相关性：cosine(h_t, h_s) ≥ θ（用归一化向量）
        h_s_norm = h_s / (np.linalg.norm(h_s) + epsilon)
        cos_sim = np.dot(h_t_norm, h_s_norm)
        if cos_sim < sim_threshold:
            continue
        
        # 3.2 线性无关性：Gram(h_t, h_s) > δ（用归一化向量）
        # Gram(a,b) = ‖a‖²₂·‖b‖²₂ - (aᵀb)²
        norm_h_t = np.linalg.norm(h_t_norm)
        norm_h_s = np.linalg.norm(h_s_norm)
        dot_product = np.dot(h_t_norm, h_s_norm)
        gram_det = (norm_h_t ** 2) * (norm_h_s ** 2) - (dot_product ** 2)
        
        if gram_det > gram_threshold:
            S_eff.append(h_s_norm)
            S_eff_original.append(h_s)  # 保存原始向量
            S_eff_info.append({
                'idx': global_idx,
                'cosine': cos_sim,
                'gram': gram_det
            })
    
    # 如果没有通过双筛选得到任何有效向量，使用回退策略：
    # 在局部窗口中选择与 h_t 余弦相似度最高的向量（排除自身）作为备选 h*，
    # 以避免所有 K(t) 为 0 的情况（尤其是 sim_threshold 设得偏高时）。
    if len(S_eff) < 1:
        best_cos = -1.0
        best_vec_norm = None
        best_vec_orig = None
        best_idx = -1
        for idx, h_s in enumerate(local_window):
            global_idx = start_idx + idx
            if global_idx == token_idx:
                continue
            h_s_norm = h_s / (np.linalg.norm(h_s) + epsilon)
            cos_sim = float(np.dot(h_t_norm, h_s_norm))
            if cos_sim > best_cos:
                best_cos = cos_sim
                best_vec_norm = h_s_norm
                best_vec_orig = h_s
                best_idx = global_idx

        if best_vec_norm is not None:
            S_eff.append(best_vec_norm)
            S_eff_original.append(best_vec_orig)
            S_eff_info.append({
                'idx': best_idx,
                'cosine': best_cos,
                'gram': max(1.0 - best_cos ** 2, 0.0)
            })
        else:
            return 0.0  # 窗口内无其他向量，无法计算
    
    S_eff = np.array(S_eff)
    S_eff_original = np.array(S_eff_original)

    # 4. 选择核心上下文向量 h*_t（用归一化向量选择）
    # h*(t) = argmax_{h_s ∈ S_eff} [cosine(h_t, h_s) · Gram(h_t, h_s)]
    best_score = -float('inf')
    h_star_norm = None
    h_star_original = None
    h_star_idx = -1
    for i, h_s_norm in enumerate(S_eff):
        cos_sim = float(np.dot(h_t_norm, h_s_norm))
        gram_det = max(1.0 - cos_sim ** 2, 0.0)
        score = cos_sim * gram_det
        if score > best_score:
            best_score = score
            h_star_norm = h_s_norm
            h_star_original = S_eff_original[i]
            h_star_idx = i

    if h_star_original is None:
        print(f"Token {token_idx}: 未能选出核心向量 h*，返回0")
        return 0.0

    # 调试：打印 S_eff 大小与选中向量信息
    try:
        print(f"Token {token_idx}: |S_eff|={len(S_eff)}, selected_h*_idx={h_star_idx}, best_score={best_score:.6e}")
    except Exception:
        pass

    # 按用户确认的实现：s 为向量维度索引（按维度 D 计算样本协方差/方差）
    h_t_vec = h_t_np.astype(float)
    h_star_vec = h_star_original.astype(float)

    if unit_normalize:
        # 单位化（用于核心项计算）
        norm_ht = np.linalg.norm(h_t_vec) + epsilon
        norm_hs = np.linalg.norm(h_star_vec) + epsilon
        h_t_unit = h_t_vec / norm_ht
        h_star_unit = h_star_vec / norm_hs

        D = h_t_unit.shape[0]
        mean_ht = float(np.mean(h_t_unit))
        mean_hs = float(np.mean(h_star_unit))
        if D > 1:
            cov_h_t_h_star = float(np.sum((h_t_unit - mean_ht) * (h_star_unit - mean_hs)) / (D - 1))
            var_h_t = float(np.sum((h_t_unit - mean_ht) ** 2) / (D - 1))
        else:
            cov_h_t_h_star = 0.0
            var_h_t = epsilon

        # Gram 与内积在单位化向量上计算，确保 Gram ∈ [0,1]
        dot_h_t_h_star = float(np.dot(h_t_unit, h_star_unit))
        norm_h_t_sq = float(np.dot(h_t_unit, h_t_unit))
        norm_h_star_sq = float(np.dot(h_star_unit, h_star_unit))
        gram_determinant = max(norm_h_t_sq * norm_h_star_sq - (dot_h_t_h_star ** 2), gram_threshold)
    else:
        # 使用原始向量进行统计量计算（不归一化）
        D = h_t_vec.shape[0]
        mean_ht = float(np.mean(h_t_vec))
        mean_hs = float(np.mean(h_star_vec))
        if D > 1:
            cov_h_t_h_star = float(np.sum((h_t_vec - mean_ht) * (h_star_vec - mean_hs)) / (D - 1))
            var_h_t = float(np.sum((h_t_vec - mean_ht) ** 2) / (D - 1))
        else:
            cov_h_t_h_star = 0.0
            var_h_t = epsilon

        dot_h_t_h_star = float(np.dot(h_t_vec, h_star_vec))
        norm_h_t_sq = float(np.dot(h_t_vec, h_t_vec))
        norm_h_star_sq = float(np.dot(h_star_vec, h_star_vec))
        gram_determinant = max(norm_h_t_sq * norm_h_star_sq - (dot_h_t_h_star ** 2), gram_threshold)

    # 计算原始曲率 K0 按照给定公式
    numerator = cov_h_t_h_star * norm_h_t_sq - dot_h_t_h_star * var_h_t
    denominator = gram_determinant * var_h_t + epsilon
    K_0 = numerator / denominator

    # 打印中间数值用于验算（使用单位化向量的统计量）
    try:
        cos_sim_display = S_eff_info[h_star_idx]['cosine'] if (0 <= h_star_idx < len(S_eff_info)) else float(dot_h_t_h_star)
        kind = 'unit' if unit_normalize else 'raw'
        print(f"Token {token_idx} DEBUG({kind}): mean_ht={mean_ht:.6e}, mean_hs={mean_hs:.6e}, cov={cov_h_t_h_star:.6e}, var={var_h_t:.6e}, dot={dot_h_t_h_star:.6e}, gram={gram_determinant:.6e}, cos={cos_sim_display:.6f}")
    except Exception:
        pass
    
    # 8. 获取形态-语义匹配度 M(t)
    if precomputed_m_t is not None:
        M_t = np.clip(precomputed_m_t, 0.0, 1.0)
    else:
        M_t = 0.5  # 默认值
    
    # 9. 计算修正因子 γ(M(t)) = 1 + α·sign(K₀)·M(t)
    sign_K0 = np.sign(K_0)
    gamma_M_t = 1.0 + alpha * sign_K0 * M_t
    
    # 10. 最终曲率
    K_t = K_0 * gamma_M_t
    
    # 详细调试输出
    print(f"\n{'='*80}")
    print(f"Token {token_idx} 曲率计算详情:")
    print(f"{'-'*80}")
    print(f"【局部上下文】")
    print(f"  窗口范围: [{start_idx}, {end_idx}]")
    print(f"  有效向量数 |S_eff|: {len(S_eff_original)}")
    print(f"  窗口大小 k: {window_size}")
    print(f"  语义阈值 θ: {sim_threshold}")
    print(f"  Gram阈值 δ: {gram_threshold}")
    print(f"\n【有效向量集合 S_eff】")
    for i, info in enumerate(S_eff_info):
        marker = " ← 核心向量h*" if i == h_star_idx else ""
        print(f"  [{i}] Token {info['idx']:2d}  |  cosine={info['cosine']:.4f}  |  Gram={info['gram']:.6e}  |  score={info['cosine']*info['gram']:.6e}{marker}")
    print(f"\n【核心上下文向量 h*】")
    if h_star_idx >= 0:
        h_star_info = S_eff_info[h_star_idx]
        print(f"  Token索引: {h_star_info['idx']}")
        print(f"  余弦相似度: {h_star_info['cosine']:.6f}")
        print(f"  Gram行列式: {h_star_info['gram']:.6e}")
        print(f"  选择得分 (cosine×Gram): {best_score:.6e}")
    
    print(f"\n【几何统计参数】")
    print(f"  几何协方差 Cov(h,h*): {cov_h_t_h_star:.6f} (夹角余弦)")
    print(f"  几何方差 Var(h): {var_h_t:.6f} (1-平均相似度)")
    print(f"  范数平方 ||h||^2: {norm_h_t_sq:.6f}")
    print(f"  内积 h·h*: {dot_h_t_h_star:.6f}")
    print(f"  Gram行列式: {gram_determinant:.6e}")
    print(f"\n【曲率计算】")
    print(f"  分子 = Cov·||h||^2 - (h·h*)·Var = {cov_h_t_h_star:.6f}×{norm_h_t_sq:.6f} - {dot_h_t_h_star:.6f}×{var_h_t:.6f}")
    print(f"       = {cov_h_t_h_star * norm_h_t_sq:.6f} - {dot_h_t_h_star * var_h_t:.6f} = {numerator:.6f}")
    print(f"  分母 = Gram·Var + ε = {gram_determinant:.6e}×{var_h_t:.6f} + {epsilon} = {denominator:.6e}")
    print(f"  K₀(t) = 分子/分母 = {K_0:.6f}")
    print(f"\n【形态-语义修正】")
    print(f"  M(t) 形态匹配度: {M_t:.6f}")
    print(f"  α 权重系数: {alpha}")
    print(f"  sign(K₀): {sign_K0:+.0f}")
    print(f"  γ(M(t)) = 1 + α·sign(K₀)·M(t) = 1 + {alpha}×{sign_K0:+.0f}×{M_t:.6f} = {gamma_M_t:.6f}")
    print(f"\n【最终结果】")
    print(f"  K(t) = K₀×γ = {K_0:.6f} × {gamma_M_t:.6f} = {K_t:.6f}")
    print(f"{'='*80}\n")
    
    return float(K_t)

# 批量计算函数（适配新增参数）
def compute_curvature_batch(
    hidden_states: torch.Tensor,
    sentence_length: int,
    window_size: int = 3,
    sim_threshold: float = 0.3,
    precomputed_m_t_list: Optional[list] = None,  # 批量传入M(t)
    unit_normalize: bool = True
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
            precomputed_m_t=m_t,
            unit_normalize=unit_normalize
        )
        curvatures.append(k_t)
    return curvatures


def compute_curvature_batch_with_scaled(
    hidden_states: torch.Tensor,
    sentence_length: int,
    window_size: int = 3,
    sim_threshold: float = 0.3,
    precomputed_m_t_list: Optional[list] = None,
    unit_normalize: bool = True,
    scale_by_D: bool = True
) -> tuple:
    """批量计算并返回原始 K 列表与缩放后的 K 列表（向后兼容）

    返回: (k_list, k_scaled_list)
    如果 scale_by_D=True，则 k_scaled = k * D（D 为 embedding 维度）
    """
    k_list = compute_curvature_batch(
        hidden_states=hidden_states,
        sentence_length=sentence_length,
        window_size=window_size,
        sim_threshold=sim_threshold,
        precomputed_m_t_list=precomputed_m_t_list,
        unit_normalize=unit_normalize
    )

    D = int(hidden_states.shape[1]) if hidden_states.ndim >= 2 else 1
    if scale_by_D:
        k_scaled = [float(k * D) for k in k_list]
    else:
        k_scaled = list(k_list)

    return k_list, k_scaled