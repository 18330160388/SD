import torch
import numpy as np
from scipy.special import gamma as gamma_func
from typing import List, Optional
try:
    from ltp import LTP
    from cnradical import Radical, RunOption
    LTP_AVAILABLE = True
except ImportError:
    LTP_AVAILABLE = False
    print("Warning: LTP or cnradical not available. Please install: pip install ltp cnradical")

# 全局LTP实例（避免重复加载）
_ltp_instance = None
_radical_extractor = None

def get_ltp_instance():
    """获取LTP单例"""
    global _ltp_instance
    if _ltp_instance is None and LTP_AVAILABLE:
        _ltp_instance = LTP()
    return _ltp_instance

def get_radical_extractor():
    """获取部首提取器单例"""
    global _radical_extractor
    if _radical_extractor is None and LTP_AVAILABLE:
        _radical_extractor = Radical(RunOption.Radical)
    return _radical_extractor

def calculate_morph_semantic_match(h_t: torch.Tensor, token_text: str, tokenizer) -> float:
    """
    计算形态-语义匹配度 M(t)（适配中文特性）
    基于字符部首特征与语义向量的相似度
    使用LTP进行词性分析，cnradical进行部首提取
    """
    if not LTP_AVAILABLE:
        return 0.5  # 降级到默认值
    
    # 使用cnradical提取当前token的部首
    def get_radical(char: str) -> Optional[str]:
        radical_extractor = get_radical_extractor()
        if radical_extractor is None:
            return None
        try:
            # 获取康熙部首
            radical_info = radical_extractor.trans_ch(char)
            return radical_info if radical_info else None
        except Exception:
            return None
    
    # 筛选单字符token（中文形态特征主要体现在单字）
    if len(token_text) == 1 and '\u4e00' <= token_text <= '\u9fff':  # 判断是否为中文字符
        radical = get_radical(token_text)
        if radical:
            # 使用LTP获取词性信息辅助语义分析
            ltp = get_ltp_instance()
            
            # 构建同部首字符集（从常用字库中筛选）
            radical_extractor = get_radical_extractor()
            radical_chars = []
            
            # 常用汉字范围采样（优化：可预先构建部首-汉字索引）
            common_chars_sample = ["海", "河", "湖", "江", "松", "柏", "树", "林", 
                                  "炎", "烧", "灯", "火", "吃", "喝", "叫", "口",
                                  "打", "拍", "提", "扌", "想", "情", "心", "忄"]
            
            for char in common_chars_sample:
                try:
                    if radical_extractor and radical_extractor.trans_ch(char) == radical:
                        radical_chars.append(char)
                        if len(radical_chars) >= 3:  # 取3个同部首字符
                            break
                except Exception:
                    continue
            
            if len(radical_chars) >= 2:  # 至少需要2个同部首字符才有意义
                # 获取同部首字符的语义向量（作为形态基准）
                radical_inputs = tokenizer(
                    radical_chars, return_tensors="pt", padding=True, truncation=True
                ).to(h_t.device)
                with torch.no_grad():
                    # 用同模型提取同部首字符的语义向量（取与h_t同层）
                    from llm_hidden_extractor import extract_hidden_states
                    radical_hidden, _, _, _, _ = extract_hidden_states(
                        text=" ".join(radical_chars),
                        middle_layer_idx=0,  # 实际应与h_t所在层一致，此处需外部传入
                        device=h_t.device
                    )
                # 计算h_t与同部首向量的平均相似度
                h_t_norm = h_t / (torch.norm(h_t) + 1e-8)
                radical_norm = radical_hidden / (torch.norm(radical_hidden, dim=1, keepdim=True) + 1e-8)
                avg_sim = torch.mean(torch.matmul(radical_norm, h_t_norm.unsqueeze(-1))).item()
                return min(max(avg_sim, 0.0), 1.0)  # 归一化到[0,1]
    
    return 0.5  # 非单字token或无部首信息时默认中等匹配度

def compute_c_t(
    h_t: torch.Tensor,
    hidden_states: torch.Tensor,
    token_idx: int,
    k: int = 3,  # 上下文窗口大小（适配中文短距离依赖）
    theta: float = 0.5,  # 语义相似度阈值
    alpha: float = 0.4,  # 形态修正因子权重
    eps: float = 1e-6,
    precomputed_m_t: Optional[float] = None  # 预计算的 M(t) 值
) -> float:
    """
    计算第t个token的局部语义聚类密度 C(t)
    严格遵循文档定义：C(t) = [N_eff(t) / (V_local(t)+eps)] * γ(M(t))
    
    参数:
        precomputed_m_t: 预计算的形态-语义匹配度 M(t)（来自 m_t_calculator）
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
        
        # 计算椭圆体体积：V = (π^(d/2) / Γ(d/2+1)) * sqrt(det(Σ))
        d = h_t.shape[0]  # 语义向量维度
        det_cov = torch.det(cov_matrix).item() if d > 1 else 1.0
        det_cov = max(det_cov, eps)  # 避免负行列式或零行列式
        
        # 伽马函数 Γ(d/2+1)
        gamma_value = gamma_func(d/2 + 1)
        volume_coeff = np.power(np.pi, d/2) / gamma_value
        V_local = volume_coeff * np.sqrt(det_cov)
        V_local = max(V_local, eps)  # 确保体积为正

    # 4. 计算中文形态-语义修正因子 γ(M(t))
    # 使用预计算的 M(t)（与 K(t)、D(t) 保持一致）
    if precomputed_m_t is not None:
        M_t = np.clip(precomputed_m_t, 0.0, 1.0)
    else:
        M_t = 0.5  # 降级默认值（中等匹配度）
    
    gamma_M = 1 + alpha * M_t  # 单调递增函数：M(t)↑ → γ↑ → C(t)↑

    # 5. 计算最终C(t)
    C_t = (N_eff / (V_local + eps)) * gamma_M
    return float(C_t)

def compute_c_t_batch(
    hidden_states: torch.Tensor,
    k: int = 3,
    theta: float = 0.5,
    alpha: float = 0.4,
    precomputed_m_t_list: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    批量计算所有token的C(t)
    
    参数:
        hidden_states: 隐藏状态张量 (seq_len, d)
        k: 上下文窗口大小
        theta: 语义相似度阈值
        alpha: 形态修正因子权重
        precomputed_m_t_list: 预计算的 M(t) 数组 (seq_len,)
    """
    seq_len = hidden_states.shape[0]
    C_t_list = []
    
    for token_idx in range(seq_len):
        h_t = hidden_states[token_idx]
        precomputed_m_t = precomputed_m_t_list[token_idx] if precomputed_m_t_list is not None else None
        
        C_t = compute_c_t(
            h_t=h_t,
            hidden_states=hidden_states,
            token_idx=token_idx,
            k=k,
            theta=theta,
            alpha=alpha,
            precomputed_m_t=precomputed_m_t
        )
        C_t_list.append(C_t)
    
    return np.array(C_t_list)