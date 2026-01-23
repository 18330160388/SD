import numpy as np
import torch
from llm_hidden_extractor import extract_hidden_states

# =====================
# 端到端语义曲率分析：向量提取 → K(t)计算 → Z-score归一化 → 聚集/发散分析
# =====================

def compute_covariance_matrix(window_vectors):
    """
    计算窗口内向量的协方差矩阵 Σ(t)
    参数：
        window_vectors: 列表，包含窗口内的向量 (N, hidden_dim)
    返回：
        cov_matrix: (hidden_dim, hidden_dim) 的协方差矩阵
    """
    if len(window_vectors) == 0:
        return np.zeros((896, 896))  # 默认896维
    vectors = np.array(window_vectors)  # (N, hidden_dim)
    # 中心化
    mean_vec = np.mean(vectors, axis=0)  # (hidden_dim,)
    centered = vectors - mean_vec  # (N, hidden_dim)
    # 协方差矩阵
    cov_matrix = np.cov(centered.T, ddof=1)  # (hidden_dim, hidden_dim)
    return cov_matrix

def compute_k_t(cov_matrix, m_t=0.5, epsilon=1e-8, zeta=0.3):
    """
    计算原始曲率 K(t)
    K(t) = [det(Σ) + ε] / [(tr(Σ))² * (1 - ζ·M(t))²]
    """
    det_cov = np.linalg.det(cov_matrix)
    trace_cov = np.trace(cov_matrix)
    denominator = (trace_cov ** 2) * ((1 - zeta * m_t) ** 2)
    if denominator == 0:
        return 0.0
    k_t = (det_cov + epsilon) / denominator
    return k_t

def compute_k_log_t(cov_matrix, m_t=0.5, epsilon=1e-8, zeta=0.3):
    """
    计算对数曲率 K_log(t)
    K_log(t) = ln(det(Σ) + ε) - 2*ln(tr(Σ) + ε) - 2*ln|1 - ζ·M(t)|
    """
    det_cov = np.linalg.det(cov_matrix)
    trace_cov = np.trace(cov_matrix)
    term1 = np.log(det_cov + epsilon)
    term2 = 2 * np.log(trace_cov + epsilon)
    term3 = 2 * np.log(abs(1 - zeta * m_t))
    k_log_t = term1 - term2 - term3
    return k_log_t

def z_score_normalize(k_log_list):
    """
    Z-score归一化 K_log(t) 到 K_norm(t)
    """
    if not k_log_list:
        return []
    mu = np.mean(k_log_list)
    sigma = np.std(k_log_list, ddof=1)
    if sigma == 0:
        return [0.0] * len(k_log_list)
    return [(k_log - mu) / sigma for k_log in k_log_list]

def compute_semantic_curvature(text, model_name="D:\\liubotao\\other\\BIT_TS\\LLM_GCG\\code\\models\\Qwen2___5-0___5B-Instruct", layer_idx=12, k=2, m_t=0.5, epsilon=1e-8, zeta=0.3):
    """
    通用函数：计算句子的语义曲率K(t)、K_log(t)、K_norm(t)
    参数：
        text: 输入句子
        model_name: 模型路径
        layer_idx: 层索引，默认12
        k: 窗口半宽，默认2 (N=5)
        m_t: M(t)默认值，默认0.5
        epsilon: 正则化参数，默认1e-8
        zeta: ζ参数，默认0.3
    返回：
        results: 列表，每个元素为字典{"token": str, "k_t": float, "k_log_t": float, "k_norm_t": float, "interpretation": str}
    """
    # 从text提取
    try:
        h_t, token_num, tokenizer, inputs, _ = extract_hidden_states(text, model_name=model_name, middle_layer_idx=layer_idx)
        h_t = h_t.numpy()
        hidden_dim = h_t.shape[1]
        # 获取token列表
        tokens = []
        for token_id in inputs['input_ids'][0]:
            token_text = tokenizer.decode([token_id])
            if token_text not in ['<|endoftext|>', '<|im_start|>', '<|im_end|>']:
                tokens.append(token_text)
        if len(tokens) != token_num:
            tokens = [f"token_{i}" for i in range(token_num)]
    except Exception as e:
        raise RuntimeError(f"向量提取失败: {e}")

    # 计算每个token的曲率
    N = 2 * k + 1
    results = []

    for t in range(token_num):
        # 窗口向量
        window_indices = list(range(max(0, t - k), min(token_num, t + k + 1)))
        window_vectors = [h_t[idx] for idx in window_indices]
        while len(window_vectors) < N:
            window_vectors.append(np.zeros(hidden_dim))

        # 协方差
        cov_matrix = compute_covariance_matrix(window_vectors)

        # K(t)
        k_t = compute_k_t(cov_matrix, m_t, epsilon, zeta)

        # K_log(t)
        k_log_t = compute_k_log_t(cov_matrix, m_t, epsilon, zeta)

        results.append({
            'token': tokens[t] if t < len(tokens) else f'token_{t}',
            'k_t': k_t,
            'k_log_t': k_log_t
        })

    # Z-score归一化
    k_log_list = [r['k_log_t'] for r in results]
    k_norm_list = z_score_normalize(k_log_list)

    for i, r in enumerate(results):
        r['k_norm_t'] = k_norm_list[i]
        if r['k_norm_t'] > 0:
            r['interpretation'] = "语义聚集"
        elif r['k_norm_t'] < 0:
            r['interpretation'] = "语义发散"
        else:
            r['interpretation'] = "语义无差异"

    return results

def analyze_semantic_curvature(text="小猫追着蝴蝶跑过花园"):
    """
    端到端分析并打印结果
    """
    print(f"分析句子：{text}")
    print("=" * 80)

    try:
        results = compute_semantic_curvature(text)
        print(f"向量提取成功：{len(results)} 个token，896 维向量")
    except Exception as e:
        print(f"错误: {e}")
        return

    # 输出表格
    print(f"{'Token':<10} {'K(t)':<12} {'K_log(t)':<12} {'K_norm(t)':<12} {'语义解读'}")
    print("-" * 80)
    for r in results:
        print(f"{r['token']:<10} {r['k_t']:<12.6f} {r['k_log_t']:<12.3f} {r['k_norm_t']:<12.3f} {r['interpretation']}")

    print("\n语义分析解释：")
    print("- 句子描述动态场景，动作词可能显示语义发散（变化/运动），名词显示聚集（稳定实体）。")
    print("- K_norm(t)正值表示相对聚集，负值表示相对发散，基于整句话的Z-score分布。")

if __name__ == "__main__":
    analyze_semantic_curvature()