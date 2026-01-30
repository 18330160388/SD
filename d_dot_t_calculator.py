"""
平均欧氏距离变化流率 Ḋ(t) 计算模块
基于文档公式(12)和(12-1)

核心功能：
1. 计算基础距离流率 Ḋ(t) = D(t) - D(t-1)
2. 引入上下文驱动力G(t)的修正：Ḋ_drive(t) = Ḋ(t) · (1 - σ(||G(t)||₂))
3. 物理意义：刻画语义向量分散程度的变化速率

接口设计：
- 接收单个位置的D(t), D(t-1), G(t)值，不接收整个序列
- D(t)由外部d_t_calculator预先计算
- 避免冗余计算
"""

import torch
import numpy as np
from typing import List, Optional, Dict


def compute_d_dot_t(
    D_t: float,
    D_t_minus_1: float
) -> float:
    """
    计算基础距离流率 Ḋ(t) = D(t) - D(t-1)
    
    根据文档公式(12)：
    - Ḋ(t) > 0 表示语义向量分散性增强
    - Ḋ(t) < 0 表示聚集性增强
    
    Args:
        D_t: 当前时刻的D(t)值
        D_t_minus_1: 前一时刻的D(t-1)值
    
    Returns:
        d_dot_t: 距离流率标量值
    """
    return D_t - D_t_minus_1


def compute_d_dot_drive_t(
    D_t: float,
    D_t_minus_1: float,
    G_t: torch.Tensor,
    sigma_func: str = 'sigmoid'
) -> Dict[str, float]:
    """
    计算驱动力修正的距离流率 Ḋ_drive(t)
    
    根据文档公式(12-1)：
    Ḋ_drive(t) = Ḋ(t) · (1 - σ(||G(t)||₂))
    
    物理意义：
    - ||G(t)||₂ 越大，上下文约束越强，(1-σ(||G(t)||₂)) 越小，距离流率增幅被抑制
    - 驱动力抑制距离发散的作用
    
    Args:
        D_t: 当前时刻的D(t)值
        D_t_minus_1: 前一时刻的D(t-1)值
        G_t: 当前时刻的上下文驱动力向量 (hidden_dim,)
        sigma_func: σ函数类型，'sigmoid'或'tanh'
    
    Returns:
        result: 字典，包含：
            - 'd_dot_t': 基础流率 Ḋ(t)
            - 'g_t_norm': 驱动力范数 ||G(t)||₂
            - 'sigma_g': σ(||G(t)||₂)
            - 'd_dot_drive_t': 修正流率 Ḋ_drive(t)
    """
    # 1. 计算基础流率 Ḋ(t)
    d_dot_t = compute_d_dot_t(D_t, D_t_minus_1)
    
    # 2. 计算驱动力范数 ||G(t)||₂
    g_t_norm = torch.norm(G_t, p=2).item()
    
    # 3. 应用σ函数
    if sigma_func == 'sigmoid':
        sigma_g = 1.0 / (1.0 + np.exp(-g_t_norm))
    elif sigma_func == 'tanh':
        sigma_g = np.tanh(g_t_norm)
    else:
        raise ValueError(f"不支持的σ函数类型: {sigma_func}")
    
    # 4. 计算修正流率 Ḋ_drive(t) = Ḋ(t) · (1 - σ(||G(t)||₂))
    d_dot_drive_t = d_dot_t * (1.0 - sigma_g)
    
    return {
        'd_dot_t': float(d_dot_t),
        'g_t_norm': float(g_t_norm),
        'sigma_g': float(sigma_g),
        'd_dot_drive_t': float(d_dot_drive_t)
    }
