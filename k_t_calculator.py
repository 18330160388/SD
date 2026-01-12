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
    precomputed_m_t: Optional[float] = None
) -> float:
    """
    严格按照文档定义计算局部截面曲率K(t)
    
    K(t) = [Cov(h(t),h*(t))·‖h(t)‖²₂ - (h(t)ᵀh*(t))·Cov(h(t),h(t))] / 
           [Gram(h(t),h*(t))·Var(h(t)) + ε] · γ(M(t))
    
    注意：协方差和方差用原始向量计算，不用归一化向量
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
    
    if len(S_eff) < 1:
        return 0.0  # 无有效向量
    
    S_eff = np.array(S_eff)
    S_eff_original = np.array(S_eff_original)
    
    # 4. 选择核心上下文向量 h*_t（用归一化向量选择）
    # h*(t) = argmax_{h_s ∈ S_eff} [cosine(h_t, h_s) · Gram(h_t, h_s)]
    best_score = -float('inf')
    h_star_norm = None
    h_star_original = None
    h_star_idx = -1
    
    for i, h_s_norm in enumerate(S_eff):
        cos_sim = np.dot(h_t_norm, h_s_norm)
        norm_h_t = np.linalg.norm(h_t_norm)
        norm_h_s = np.linalg.norm(h_s_norm)
        gram_det = (norm_h_t ** 2) * (norm_h_s ** 2) - (cos_sim ** 2)
        
        score = cos_sim * gram_det
        if score > best_score:
            best_score = score
            h_star_norm = h_s_norm
            h_star_original = S_eff_original[i]
            h_star_idx = i
    
    if h_star_original is None:
        return 0.0
    
    # 5. 计算协方差和方差（使用归一化向量以控制数值尺度）
    # 文档要求："通过向量归一化（|h_t|_2=1，|h*_t|_2=1）将曲率核心项范围压缩至 [-1,1]"
    # 使用归一化向量确保 Cov、Var、Gram 在合理的数值范围内
    
    d = h_t_norm.shape[0]  # 向量维度
    
    # 计算归一化向量的均值（标量）
    h_t_mean = np.mean(h_t_norm)
    h_star_mean = np.mean(h_star_norm)
    
    # 协方差：各维度偏差的乘积之和 / (d-1)
    cov_h_t_h_star = np.sum((h_t_norm - h_t_mean) * (h_star_norm - h_star_mean)) / max(d - 1, 1)
    
    # Var(h_t)：h_t 各维度偏差的平方和 / (d-1)
    var_h_t = np.sum((h_t_norm - h_t_mean) ** 2) / max(d - 1, 1)
    
    # 避免方差为0
    if var_h_t < epsilon:
        var_h_t = epsilon
    
    # 6. 计算Gram行列式和内积（使用归一化向量）
    norm_h_t_sq = np.dot(h_t_norm, h_t_norm)  # ≈ 1
    norm_h_star_sq = np.dot(h_star_norm, h_star_norm)  # ≈ 1
    dot_h_t_h_star = np.dot(h_t_norm, h_star_norm)
    gram_determinant = norm_h_t_sq * norm_h_star_sq - (dot_h_t_h_star ** 2)
    gram_determinant = max(abs(gram_determinant), gram_threshold)  # 取绝对值避免负数
    
    # 7. 计算原始曲率 K₀(t)（未修正）
    # 严格按照文档定义：Cov(h,h*)·‖h‖² - (h·h*)·Var(h)
    # 其中 Cov(h(t),h(t)) = Var(h(t))
    numerator = cov_h_t_h_star * norm_h_t_sq - dot_h_t_h_star * var_h_t
    denominator = gram_determinant * var_h_t + epsilon
    K_0 = numerator / denominator
    
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
    
    print(f"\n【统计参数】")
    print(f"  协方差 Cov(h,h*): {cov_h_t_h_star:.6f}")
    print(f"  方差 Var(h): {var_h_t:.6f}")
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