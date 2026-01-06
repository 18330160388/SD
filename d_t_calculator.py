import torch
import numpy as np
from typing import Optional

def compute_d_t(
    h_t: torch.Tensor,
    hidden_states: torch.Tensor,
    token_idx: int,
    sentence_length: int,
    distance_type: str = "euclidean",
    normalize: bool = True,
    eps: float = 1e-8
) -> float:
    """
    计算Token级别的语义分布距离D(t)
    物理意义：当前token隐藏状态相对于整句所有token的"离散程度"，值越大表示该token语义越独立
    
    Args:
        h_t: 当前token的隐藏状态向量，shape=(hidden_dim,)
        hidden_states: 整句话的隐藏状态，shape=(token_num, hidden_dim)
        token_idx: 当前token的索引
        sentence_length: 句子的token总数
        distance_type: 距离计算类型，支持 "euclidean"(欧氏距离) / "cosine"(余弦距离)
        normalize: 是否对距离做归一化（缩放到0~1区间）
        eps: 防止除零的小值
    
    Returns:
        D_t: 语义分布距离D(t)，标量值
    """
    # 1. 排除当前token（避免自身距离为0的干扰）
    mask = torch.ones(sentence_length, dtype=bool)
    mask[token_idx] = False
    valid_hidden = hidden_states[mask]  # (token_num-1, hidden_dim)
    
    # 2. 计算当前token与其他所有token的距离
    if distance_type == "euclidean":
        # 欧氏距离：L2范数
        distances = torch.norm(valid_hidden - h_t.unsqueeze(0), dim=1)
    elif distance_type == "cosine":
        # 余弦距离：1 - 余弦相似度（值域0~2，越大越不相似）
        h_t_norm = h_t / (torch.norm(h_t) + eps)
        valid_norm = valid_hidden / (torch.norm(valid_hidden, dim=1, keepdim=True) + eps)
        cos_sim = torch.matmul(valid_norm, h_t_norm.unsqueeze(-1)).squeeze(-1)
        distances = 1 - cos_sim  # 余弦距离
    else:
        raise ValueError(f"不支持的距离类型：{distance_type}，仅支持 euclidean / cosine")
    
    # 3. 计算D(t)核心值：所有距离的均值（代表全局分布距离）
    D_t = torch.mean(distances).item()
    
    # 4. 归一化（缩放到0~1）
    if normalize:
        # 全局最大/最小距离（用于归一化）
        all_distances = torch.norm(hidden_states.unsqueeze(0) - hidden_states.unsqueeze(1), dim=2)
        global_max = torch.max(all_distances).item() + eps
        global_min = torch.min(all_distances).item()
        D_t = (D_t - global_min) / (global_max - global_min)
    
    return D_t


def compute_d_t_batch(
    hidden_states: torch.Tensor,
    distance_type: str = "euclidean",
    normalize: bool = True,
    eps: float = 1e-8
) -> np.ndarray:
    """
    批量计算所有token的D(t)（效率更高）
    
    Args:
        hidden_states: 整句话的隐藏状态，shape=(token_num, hidden_dim)
        distance_type: 距离计算类型
        normalize: 是否归一化
        eps: 防止除零的小值
    
    Returns:
        D_t_batch: 所有token的D(t)数组，shape=(token_num,)
    """
    token_num = hidden_states.shape[0]
    D_t_list = []
    
    for token_idx in range(token_num):
        h_t = hidden_states[token_idx]
        D_t = compute_d_t(
            h_t=h_t,
            hidden_states=hidden_states,
            token_idx=token_idx,
            sentence_length=token_num,
            distance_type=distance_type,
            normalize=normalize,
            eps=eps
        )
        D_t_list.append(D_t)
    
    return np.array(D_t_list)


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
        distance_type="euclidean",
        normalize=True
    )
    print(f"单个Token的D(t)值：{d_t:.6f}")
    
    # 批量计算所有token的D(t)
    d_t_batch = compute_d_t_batch(test_hidden, distance_type="cosine")
    print(f"\n所有Token的D(t)数组：")
    for idx, d in enumerate(d_t_batch):
        print(f"Token[{idx}] D(t) = {d:.6f}")