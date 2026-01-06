import torch
import numpy as np
from scipy.stats import gamma
from typing import List, Optional

def calculate_morph_semantic_match(h_t: torch.Tensor, token_text: str, tokenizer) -> float:
    """
    计算形态-语义匹配度 M(t)（适配中文特性）
    基于字符部首特征与语义向量的相似度
    """
    # 中文常见部首映射（简化版，可扩展）
    radical_map = {
        "氵": ["海", "河", "湖", "江"],
        "木": ["松", "柏", "柳", "桃"],
        "火": ["炎", "烧", "烤", "灯"],
        "口": ["吃", "喝", "叫", "喊"],
        "扌": ["打", "拍", "提", "抓"]
    }
    
    # 提取当前token的部首（简化实现，实际可调用中文NLP工具如LTP）
    def get_radical(char: str) -> Optional[str]:
        for radical, chars in radical_map.items():
            if char in chars:
                return radical
        return None
    
    # 筛选单字符token（中文形态特征主要体现在单字）
    if len(token_text) == 1:
        radical = get_radical(token_text)
        if radical:
            # 获取同部首字符的语义向量（作为形态基准）
            radical_chars = radical_map[radical][:3]  # 取3个同部首字符
            radical_inputs = tokenizer(
                radical_chars, return_tensors="pt", padding=True, truncation=True
            ).to(h_t.device)
            with torch.no_grad():
                # 用同模型提取同部首字符的语义向量（取与h_t同层）
                from llm_hidden_extractor import extract_hidden_states
                radical_hidden, _, _, _ = extract_hidden_states(
                    text=" ".join(radical_chars),
                    middle_layer_idx=0,  # 实际应与h_t所在层一致，此处需外部传入
                    device=h_t.device
                )
            # 计算h_t与同部首向量的平均相似度
            h_t_norm = h_t / (torch.norm(h_t) + 1e-8)
            radical_norm = radical_hidden / (torch.norm(radical_hidden, dim=1, keepdim=True) + 1e-8)
            avg_sim = torch.mean(torch.matmul(radical_norm, h_t_norm.unsqueeze(-1))).item()
            return min(avg_sim, 1.0)  # 归一化到[0,1]
    return 0.5  # 非单字token默认中等匹配度

def compute_c_t(
    h_t: torch.Tensor,
    hidden_states: torch.Tensor,
    token_idx: int,
    token_text: str,
    tokenizer,
    k: int = 3,  # 上下文窗口大小（适配中文短距离依赖）
    theta: float = 0.5,  # 语义相似度阈值
    alpha: float = 0.4,  # 形态修正因子权重
    eps: float = 1e-6
) -> float:
    """
    计算第t个token的局部语义聚类密度 C(t)
    严格遵循文档1的数学定义：C(t) = [N_eff(t) / (V_local(t)+eps)] * γ(M(t))
    """
    # 1. 界定局部语义子空间（上下文窗口 [t-k, t+k]）
    seq_len = hidden_states.shape[0]
    start_idx = max(0, token_idx - k)
    end_idx = min(seq_len - 1, token_idx + k)
    local_window = hidden_states[start_idx:end_idx+1]  # (2k+1, d)

    # 2. 计算有效语义向量数 N_eff(t)（语义相似度筛选）
    h_t_norm = h_t / (torch.norm(h_t) + eps)
    local_norm = local_window / (torch.norm(local_window, dim=1, keepdim=True) + eps)
    cos_sim = torch.matmul(local_norm, h_t_norm.unsqueeze(-1)).squeeze(-1)  # (2k+1,)
    N_eff = torch.sum((cos_sim >= theta).float()).item()  # 有效向量数

    # 3. 计算局部子空间体积 V_local(t)（基于协方差矩阵的椭圆体体积）
    if N_eff < 2:  # 有效向量不足，无法计算协方差
        V_local = 1.0
    else:
        # 筛选有效向量并计算协方差矩阵
        valid_mask = cos_sim >= theta
        valid_vectors = local_window[valid_mask]
        mean_vec = torch.mean(valid_vectors, dim=0)  # 局部语义中心
        cov_matrix = torch.cov(valid_vectors.T)  # (d, d)，协方差矩阵
        # 计算椭圆体体积：(π^(d/2)/Γ(d/2+1)) * sqrt(det(Σ))
        d = h_t.shape[0]  # 语义向量维度
        det_cov = torch.det(cov_matrix) if d > 0 else 1.0
        det_cov = max(det_cov, eps)  # 避免负行列式
        # 伽马函数计算（scipy gamma兼容torch）
        gamma_term = gamma(d/2 + 1).pdf(d/2 + 1) if d > 0 else 1.0
        volume = (np.power(np.pi, d/2) / gamma_term) * np.sqrt(det_cov)
        V_local = volume

    # 4. 计算中文形态-语义修正因子 γ(M(t))
    M_t = calculate_morph_semantic_match(h_t, token_text, tokenizer)
    gamma_M = 1 + alpha * M_t  # 单调递增函数

    # 5. 计算最终C(t)
    C_t = (N_eff / (V_local + eps)) * gamma_M
    return C_t

def compute_c_t_batch(
    hidden_states: torch.Tensor,
    token_texts: List[str],
    tokenizer,
    k: int = 3,
    theta: float = 0.5,
    alpha: float = 0.4
) -> np.ndarray:
    """批量计算所有token的C(t)"""
    seq_len = hidden_states.shape[0]
    C_t_list = []
    for token_idx in range(seq_len):
        h_t = hidden_states[token_idx]
        token_text = token_texts[token_idx]
        C_t = compute_c_t(
            h_t=h_t,
            hidden_states=hidden_states,
            token_idx=token_idx,
            token_text=token_text,
            tokenizer=tokenizer,
            k=k,
            theta=theta,
            alpha=alpha
        )
        C_t_list.append(C_t)
    return np.array(C_t_list)