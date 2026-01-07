import torch
import numpy as np
from typing import List, Optional

def compute_v_local(h_t: torch.Tensor, hidden_states: torch.Tensor, token_idx: int, k: int = 2) -> float:
    """计算局部上下文一致性势能 V_local（文档2公式）"""
    seq_len = hidden_states.shape[0]
    start_idx = max(0, token_idx - k)
    end_idx = min(seq_len - 1, token_idx + k)
    local_window = hidden_states[start_idx:end_idx+1]
    # 排除自身token
    local_window = torch.cat([local_window[:token_idx-start_idx], local_window[token_idx-start_idx+1:]])
    if local_window.shape[0] == 0:
        return 1.0  # 无上下文时势能最高
    # 计算余弦相似度，使用固定分母 1/(2k) 严格按照文档公式
    h_t_norm = h_t / (torch.norm(h_t) + 1e-8)
    local_norm = local_window / (torch.norm(local_window, dim=1, keepdim=True) + 1e-8)
    sum_sim = torch.sum(torch.matmul(local_norm, h_t_norm.unsqueeze(-1))).item()
    avg_sim = sum_sim / (2 * k)  # 固定分母2k，非实际邻居数
    return 1 - avg_sim  # 相似度越低，势能越高

def compute_v_morph(h_t: torch.Tensor, m_t: np.ndarray, morph_embedding: torch.nn.Module) -> float:
    """计算形态-语义一致性势能 V_morph（文档2公式：欧氏距离平方）
    
    Args:
        h_t: 语义编码向量（d维）
        m_t: 形态特征向量（224维，来自M(t)模块的morph_extractor）
        morph_embedding: MorphEmbedding模块（将224维m(t)映射到d维）
    
    Returns:
        V_morph = ||h(t) - MorphEmbedding(m(t))||²₂
    """
    if m_t is None:
        return 0.5  # 非中文字符默认中等势能
    
    # 确保在同一设备上
    device = h_t.device
    
    # 使用MorphEmbedding将m(t)映射到语义空间：Φ(m(t))
    phi_m_t = morph_embedding(m_t).to(device)  # (d,)
    
    # 计算欧氏距离平方：||h(t) - Φ(m(t))||²₂
    diff = h_t - phi_m_t
    v_morph = torch.sum(diff ** 2).item()
    
    return v_morph

def compute_v_hier(h_t: torch.Tensor, hidden_states: torch.Tensor, token_group: List[int]) -> float:
    """计算层级编码一致性势能 V_hier（文档2公式：1-余弦相似度）"""
    # token_group：当前token所属高层级单元的其他token索引（不含自身）
    if len(token_group) == 0:
        return 0.0  # 无高层级上下文时势能为0（完全一致）
    
    # 确保所有张量在同一设备上
    device = h_t.device
    hidden_states = hidden_states.to(device)
    group_vectors = hidden_states[token_group]
    hier_avg = torch.mean(group_vectors, dim=0)  # 高层级聚合编码 h̄(t)
    # 计算余弦相似度
    h_t_norm = h_t / (torch.norm(h_t) + 1e-8)
    hier_norm = hier_avg / (torch.norm(hier_avg) + 1e-8)
    cos_sim = torch.matmul(h_t_norm, hier_norm).item()
    return 1 - cos_sim

def compute_v_global(h_t: torch.Tensor, hidden_states: torch.Tensor, attn_weights: torch.Tensor) -> float:
    """计算全局语义锚定势能 V_global（文档2公式：归一化欧氏距离）"""
    # attn_weights：自注意力权重（shape: (seq_len,)），来自LLM真实输出
    # 确保所有张量在同一设备上
    device = h_t.device
    hidden_states = hidden_states.to(device)
    attn_weights = attn_weights.to(device)
    attn_norm = attn_weights / (torch.sum(attn_weights) + 1e-8)
    global_vec = torch.sum(hidden_states * attn_norm.unsqueeze(1), dim=0)  # 全局语义锚点
    # 归一化欧氏距离
    numerator = torch.norm(h_t - global_vec, p=2).item() ** 2
    denominator = torch.norm(global_vec, p=2).item() ** 2 + 1e-8
    return numerator / denominator

def compute_v_poly(h_t: torch.Tensor, poly_mlp: torch.nn.Module, num_senses: int = 5) -> float:
    """计算多义性消解一致性势能 V_poly（文档2公式：义项分布熵）"""
    # poly_mlp：预训练的义项分类器（输入h_t，输出各义项概率）
    device = next(poly_mlp.parameters()).device
    h_t = h_t.to(device)
    sense_logits = poly_mlp(h_t.unsqueeze(0)).squeeze(0)  # (num_senses,)
    sense_probs = torch.softmax(sense_logits, dim=0)
    # 计算熵（避免log(0)）
    sense_probs = torch.clamp(sense_probs, min=1e-8, max=1.0)
    entropy = -torch.sum(sense_probs * torch.log(sense_probs)).item()
    return entropy

def compute_v_t(
    h_t: torch.Tensor,
    hidden_states: torch.Tensor,
    token_idx: int,
    m_t: np.ndarray,
    attn_weights: torch.Tensor,
    token_group: List[int],
    morph_embedding: torch.nn.Module,
    poly_mlp: torch.nn.Module,
    lambdas: List[float] = [0.3, 0.2, 0.15, 0.2, 0.15]  # 权重系数（和为1）
) -> float:
    """
    计算语义一致性势能 V(h(t))（文档2多约束加权模型）
    """
    # 1. 计算各子势能
    v_local = compute_v_local(h_t, hidden_states, token_idx)
    v_morph = compute_v_morph(h_t, m_t, morph_embedding)
    v_hier = compute_v_hier(h_t, hidden_states, token_group)
    v_global = compute_v_global(h_t, hidden_states, attn_weights)
    v_poly = compute_v_poly(h_t, poly_mlp)
    
    # 2. 加权叠加（权重和为1）
    V_t = (
        lambdas[0] * v_local +
        lambdas[1] * v_morph +
        lambdas[2] * v_hier +
        lambdas[3] * v_global +
        lambdas[4] * v_poly
    )
    return V_t

def compute_v_t_batch(
    hidden_states: torch.Tensor,
    attn_weights: torch.Tensor,
    token_groups: List[List[int]],  # 每个token的高层级单元索引
    m_t_list: List[np.ndarray],
    morph_embedding: torch.nn.Module,
    poly_mlp: torch.nn.Module
) -> np.ndarray:
    """批量计算所有token的V(h(t))"""
    seq_len = hidden_states.shape[0]
    V_t_list = []
    for token_idx in range(seq_len):
        h_t = hidden_states[token_idx]
        m_t = m_t_list[token_idx]
        token_group = token_groups[token_idx]
        V_t = compute_v_t(
            h_t=h_t,
            hidden_states=hidden_states,
            token_idx=token_idx,
            m_t=m_t,
            attn_weights=attn_weights,
            token_group=token_group,
            morph_embedding=morph_embedding,
            poly_mlp=poly_mlp
        )
        V_t_list.append(V_t)
    return np.array(V_t_list)