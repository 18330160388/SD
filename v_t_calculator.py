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
    # 计算平均余弦相似度
    h_t_norm = h_t / (torch.norm(h_t) + 1e-8)
    local_norm = local_window / (torch.norm(local_window, dim=1, keepdim=True) + 1e-8)
    avg_sim = torch.mean(torch.matmul(local_norm, h_t_norm.unsqueeze(-1))).item()
    return 1 - avg_sim  # 相似度越低，势能越高

def compute_v_morph(h_t: torch.Tensor, token_text: str, tokenizer, morph_mlp: torch.nn.Module) -> float:
    """计算形态-语义一致性势能 V_morph（文档2公式：欧氏距离平方）"""
    # 提取形态特征（简化为部首+笔画数编码）
    def get_morph_feature(char: str) -> torch.Tensor:
        # 部首编码（one-hot，简化版）
        radical_onehot = torch.zeros(5)  # 5个常见部首
        radical_map = {"氵":0, "木":1, "火":2, "口":3, "扌":4}
        for radical, idx in radical_map.items():
            if radical in char:
                radical_onehot[idx] = 1.0
                break
        # 笔画数（简化为0-1编码，>5画为1）
        stroke_count = len(char) * 3  # 简化估算，实际需中文工具
        stroke_onehot = torch.tensor([1.0 if stroke_count > 5 else 0.0])
        return torch.cat([radical_onehot, stroke_onehot])  # (6,)
    
    if len(token_text) == 1:
        morph_feat = get_morph_feature(token_text).to(h_t.device)
        # 形态特征嵌入到语义空间
        morph_emb = morph_mlp(morph_feat.unsqueeze(0)).squeeze(0)  # (d,)
        # 欧氏距离平方
        return torch.norm(h_t - morph_emb, p=2).item() ** 2
    return 0.5  # 非单字token默认中等势能

def compute_v_hier(h_t: torch.Tensor, hidden_states: torch.Tensor, token_group: List[int]) -> float:
    """计算层级编码一致性势能 V_hier（文档2公式：1-余弦相似度）"""
    # token_group：当前token所属高层级单元（如子词）的token索引列表
    group_vectors = hidden_states[token_group]
    hier_avg = torch.mean(group_vectors, dim=0)  # 高层级聚合编码
    # 计算余弦相似度
    h_t_norm = h_t / (torch.norm(h_t) + 1e-8)
    hier_norm = hier_avg / (torch.norm(hier_avg) + 1e-8)
    cos_sim = torch.matmul(h_t_norm, hier_norm).item()
    return 1 - cos_sim

def compute_v_global(h_t: torch.Tensor, hidden_states: torch.Tensor, attn_weights: torch.Tensor) -> float:
    """计算全局语义锚定势能 V_global（文档2公式：归一化欧氏距离）"""
    # attn_weights：自注意力权重（shape: (seq_len,)），来自LLM真实输出
    attn_norm = attn_weights / (torch.sum(attn_weights) + 1e-8)
    global_vec = torch.sum(hidden_states * attn_norm.unsqueeze(1), dim=0)  # 全局语义锚点
    # 归一化欧氏距离
    numerator = torch.norm(h_t - global_vec, p=2).item() ** 2
    denominator = torch.norm(global_vec, p=2).item() ** 2 + 1e-8
    return numerator / denominator

def compute_v_poly(h_t: torch.Tensor, poly_mlp: torch.nn.Module, num_senses: int = 5) -> float:
    """计算多义性消解一致性势能 V_poly（文档2公式：义项分布熵）"""
    # poly_mlp：预训练的义项分类器（输入h_t，输出各义项概率）
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
    token_text: str,
    tokenizer,
    attn_weights: torch.Tensor,
    token_group: List[int],
    morph_mlp: torch.nn.Module,
    poly_mlp: torch.nn.Module,
    lambdas: List[float] = [0.3, 0.2, 0.15, 0.2, 0.15]  # 权重系数（和为1）
) -> float:
    """
    计算语义一致性势能 V(h(t))（文档2多约束加权模型）
    """
    # 1. 计算各子势能
    v_local = compute_v_local(h_t, hidden_states, token_idx)
    v_morph = compute_v_morph(h_t, token_text, tokenizer, morph_mlp)
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
    token_texts: List[str],
    tokenizer,
    attn_weights: torch.Tensor,
    token_groups: List[List[int]],  # 每个token的高层级单元索引
    morph_mlp: torch.nn.Module,
    poly_mlp: torch.nn.Module
) -> np.ndarray:
    """批量计算所有token的V(h(t))"""
    seq_len = hidden_states.shape[0]
    V_t_list = []
    for token_idx in range(seq_len):
        h_t = hidden_states[token_idx]
        token_text = token_texts[token_idx]
        token_group = token_groups[token_idx]
        V_t = compute_v_t(
            h_t=h_t,
            hidden_states=hidden_states,
            token_idx=token_idx,
            token_text=token_text,
            tokenizer=tokenizer,
            attn_weights=attn_weights,
            token_group=token_group,
            morph_mlp=morph_mlp,
            poly_mlp=poly_mlp
        )
        V_t_list.append(V_t)
    return np.array(V_t_list)