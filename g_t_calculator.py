"""
上下文驱动力 G(t) 计算模块
基于文档3-1《中文大语言模型语义层输入 u(t) 与上下文驱动力 g(h(t),u(t)) 的定义》

核心功能：
1. 构造多粒度输入特征 u(t)（字符+子词+形态+位置+知识）
2. 计算上下文驱动力 g(h(t), u(t))（注意力加权+门控筛选+语义融合）
3. 输出驱动语义状态更新的有效增量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


# ---------------------- 输入特征 u(t) 构造模块 ----------------------

class InputFeatureConstructor(nn.Module):
    """
    构造多粒度输入特征向量 u(t)
    u(t) = LayerNorm(Concat(e_c, e_s, e_m, e_p, e_k) · W_u + b_u)
    """
    def __init__(
        self,
        vocab_size: int = 30000,          # 字符/token词表大小
        char_dim: int = 384,              # 字符嵌入维度
        subword_dim: int = 384,           # 子词嵌入维度
        morph_dim: int = 64,              # 形态特征维度
        pos_dim: int = 64,                # 位置编码维度
        knowledge_dim: int = 64,          # 知识增强维度
        hidden_dim: int = 896,            # 输出维度（与h(t)一致）
        max_seq_len: int = 512            # 最大序列长度
    ):
        super().__init__()
        
        # 各模块维度
        self.char_dim = char_dim
        self.subword_dim = subword_dim
        self.morph_dim = morph_dim
        self.pos_dim = pos_dim
        self.knowledge_dim = knowledge_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # 总输入维度（拼接前）
        input_dim = char_dim + subword_dim + morph_dim + pos_dim + knowledge_dim
        
        # ① 字符嵌入表（预训练，这里用随机初始化模拟）
        self.char_embedding = nn.Embedding(vocab_size, char_dim)
        
        # ② 子词嵌入表（与字符共享或独立）
        self.subword_embedding = nn.Embedding(vocab_size, subword_dim)
        
        # ③ 形态特征投影（224维 → morph_dim）
        # 形态特征来自 M(t) 模块的 morph_extractor（224维）
        self.morph_projection = nn.Linear(224, morph_dim)
        
        # ④ 位置编码（相对位置编码，可学习）
        self.position_embedding = nn.Embedding(max_seq_len * 2, pos_dim)  # 支持相对位置
        
        # ⑤ 知识增强投影（简化为可学习嵌入）
        self.knowledge_embedding = nn.Embedding(vocab_size, knowledge_dim)
        
        # 线性投影：input_dim → hidden_dim
        self.projection = nn.Linear(input_dim, hidden_dim)
        
        # LayerNorm归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        token_ids: torch.Tensor,          # (seq_len,) token ID序列
        morph_features: List[np.ndarray], # List of 224-dim形态特征
        token_positions: torch.Tensor     # (seq_len,) 位置索引
    ) -> torch.Tensor:
        """
        构造输入特征向量 u(t)
        
        Args:
            token_ids: token ID序列
            morph_features: 形态特征列表（来自M(t)模块）
            token_positions: 位置索引
            
        Returns:
            u: (seq_len, hidden_dim) 输入特征向量
        """
        seq_len = token_ids.shape[0]
        device = token_ids.device
        
        # ① 字符嵌入 e_c(t)
        e_c = self.char_embedding(token_ids)  # (seq_len, char_dim)
        
        # ② 子词嵌入 e_s(t)（简化处理：与字符嵌入相同）
        e_s = self.subword_embedding(token_ids)  # (seq_len, subword_dim)
        
        # ③ 形态特征嵌入 e_m(t)
        morph_tensor_list = []
        for m_t in morph_features:
            if m_t is None:
                # 非汉字token，使用零向量
                morph_tensor_list.append(torch.zeros(224, device=device))
            else:
                morph_tensor_list.append(torch.tensor(m_t, dtype=torch.float32, device=device))
        morph_tensor = torch.stack(morph_tensor_list, dim=0)  # (seq_len, 224)
        e_m = self.morph_projection(morph_tensor)  # (seq_len, morph_dim)
        
        # ④ 位置编码 e_p(t)（相对位置编码）
        # 简化处理：使用绝对位置（实际应使用相对位置）
        e_p = self.position_embedding(token_positions)  # (seq_len, pos_dim)
        
        # ⑤ 知识增强嵌入 e_k(t)
        e_k = self.knowledge_embedding(token_ids)  # (seq_len, knowledge_dim)
        
        # 拼接所有特征
        u_concat = torch.cat([e_c, e_s, e_m, e_p, e_k], dim=-1)  # (seq_len, input_dim)
        
        # 线性投影到隐藏层维度
        u_proj = self.projection(u_concat)  # (seq_len, hidden_dim)
        
        # LayerNorm归一化
        u = self.layer_norm(u_proj)  # (seq_len, hidden_dim)
        
        return u


# ---------------------- 上下文驱动力 g(h(t), u(t)) 计算模块 ----------------------

class ContextDrivingForce(nn.Module):
    """
    计算上下文驱动力 g(h(t), u(t))
    
    三步计算：
    1. 注意力加权筛选：u_att = α(t) · u(t)
    2. 门控机制筛选：u_gate = g_gate(t) ⊙ u_att(t)
    3. 语义融合：g = GELU(W_f · Concat(h, u_gate, h⊙u_gate) + b_f)
    """
    def __init__(self, hidden_dim: int = 896, distance_decay: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.distance_decay = distance_decay  # λ: 距离衰减系数
        
        # 第一步：注意力加权
        self.W_a = nn.Linear(hidden_dim, hidden_dim)  # 注意力投影矩阵
        
        # 第二步：门控机制
        self.W_g = nn.Linear(hidden_dim * 2, hidden_dim)  # 门控权重矩阵
        
        # 第三步：语义融合
        self.W_f = nn.Linear(hidden_dim * 3, hidden_dim)  # 融合权重矩阵
        
    def forward(
        self,
        h_t: torch.Tensor,      # (hidden_dim,) 当前语义状态
        u_t: torch.Tensor,      # (hidden_dim,) 当前输入特征
        token_idx: int,         # 当前token位置
        u_all: torch.Tensor     # (seq_len, hidden_dim) 全序列输入特征（用于距离衰减）
    ) -> torch.Tensor:
        """
        计算驱动力 g(h(t), u(t))
        
        Args:
            h_t: 当前语义状态向量
            u_t: 当前输入特征向量
            token_idx: 当前token在序列中的位置
            u_all: 全序列输入特征（用于计算距离衰减）
            
        Returns:
            g_t: (hidden_dim,) 上下文驱动力向量
        """
        device = h_t.device
        seq_len = u_all.shape[0]
        
        # ==================== 第一步：注意力加权筛选 ====================
        # α(t) = softmax(h(t)^T · W_a · u(t) / √d · exp(-λ|t-s|))
        
        # 计算注意力logits
        u_projected = self.W_a(u_t)  # (hidden_dim,)
        attn_logit = torch.dot(h_t, u_projected) / np.sqrt(self.hidden_dim)  # 标量
        
        # 距离衰减（简化：仅对当前token，实际应对整个窗口）
        # 这里简化处理，实际应该计算整个上下文窗口的注意力
        # 为简化，我们仅对当前输入应用注意力权重α=1（因为只有单个输入）
        alpha = torch.sigmoid(attn_logit)  # 使用sigmoid而非softmax（单元素情况）
        
        # 注意力加权后的输入
        u_att = alpha * u_t  # (hidden_dim,)
        
        # ==================== 第二步：门控机制筛选 ====================
        # g_gate(t) = sigmoid(W_g · Concat(h(t), u_att(t)) + b_g)
        
        concat_hu = torch.cat([h_t, u_att], dim=0)  # (2*hidden_dim,)
        g_gate = torch.sigmoid(self.W_g(concat_hu))  # (hidden_dim,)
        
        # 门控筛选后的输入
        u_gate = g_gate * u_att  # (hidden_dim,) 元素-wise乘法
        
        # ==================== 第三步：语义融合与非线性激活 ====================
        # g = GELU(W_f · Concat(h, u_gate, h⊙u_gate) + b_f)
        
        h_u_interaction = h_t * u_gate  # (hidden_dim,) 元素-wise乘法，捕捉交互
        concat_final = torch.cat([h_t, u_gate, h_u_interaction], dim=0)  # (3*hidden_dim,)
        
        g_t = F.gelu(self.W_f(concat_final))  # (hidden_dim,)
        
        return g_t


# ---------------------- 完整的 G(t) 计算管线 ----------------------

class DrivingForceCalculator:
    """
    完整的上下文驱动力计算管线
    整合输入特征构造 + 驱动力计算
    """
    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_dim: int = 896,
        max_seq_len: int = 512
    ):
        self.hidden_dim = hidden_dim
        
        # 初始化输入特征构造器
        self.input_constructor = InputFeatureConstructor(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len
        )
        
        # 初始化驱动力计算器
        self.driving_force = ContextDrivingForce(hidden_dim=hidden_dim)
        
    def compute_g_t_batch(
        self,
        hidden_states: torch.Tensor,       # (seq_len, hidden_dim) 语义状态序列
        token_ids: torch.Tensor,           # (seq_len,) token ID序列
        morph_features: List[np.ndarray],  # List of 224-dim形态特征
        tokenizer                          # tokenizer对象
    ) -> np.ndarray:
        """
        批量计算所有token的驱动力 G(t)
        
        Args:
            hidden_states: 语义状态序列 h(t)
            token_ids: token ID序列
            morph_features: 形态特征列表
            tokenizer: tokenizer对象
            
        Returns:
            G_t_list: (seq_len, hidden_dim) 驱动力向量数组
        """
        seq_len = hidden_states.shape[0]
        device = hidden_states.device
        
        # 构造位置索引
        token_positions = torch.arange(seq_len, device=device)
        
        # 将模型移到正确的设备
        self.input_constructor.to(device)
        self.driving_force.to(device)
        
        # 第一步：构造全序列输入特征 u(t)
        with torch.no_grad():
            u_all = self.input_constructor(token_ids, morph_features, token_positions)
        # (seq_len, hidden_dim)
        
        # 第二步：逐token计算驱动力 g(h(t), u(t))
        G_t_list = []
        for token_idx in range(seq_len):
            h_t = hidden_states[token_idx]  # (hidden_dim,)
            u_t = u_all[token_idx]          # (hidden_dim,)
            
            with torch.no_grad():
                g_t = self.driving_force(h_t, u_t, token_idx, u_all)
            
            G_t_list.append(g_t.cpu().numpy())
        
        return np.array(G_t_list)  # (seq_len, hidden_dim)


# ---------------------- 初始化函数（方便外部调用） ----------------------

def init_driving_force_calculator(
    vocab_size: int = 30000,
    hidden_dim: int = 896,
    max_seq_len: int = 512
) -> DrivingForceCalculator:
    """
    初始化驱动力计算器
    
    Args:
        vocab_size: 词表大小
        hidden_dim: 隐藏层维度（与h(t)一致）
        max_seq_len: 最大序列长度
        
    Returns:
        calculator: 驱动力计算器实例
    """
    calculator = DrivingForceCalculator(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len
    )
    return calculator


# ---------------------- 测试代码 ----------------------

if __name__ == "__main__":
    print("=" * 60)
    print("上下文驱动力 G(t) 计算模块测试")
    print("=" * 60)
    
    # 模拟参数
    seq_len = 5
    hidden_dim = 896
    vocab_size = 30000
    
    # 初始化计算器
    calculator = init_driving_force_calculator(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim
    )
    
    # 模拟输入数据
    hidden_states = torch.randn(seq_len, hidden_dim)  # 语义状态
    token_ids = torch.randint(0, vocab_size, (seq_len,))  # token IDs
    
    # 模拟形态特征（这里用随机值，实际应来自morph_extractor）
    morph_features = [np.random.randn(224) for _ in range(seq_len)]
    
    # 模拟tokenizer（简化）
    class MockTokenizer:
        def decode(self, ids):
            return f"token_{ids[0]}"
    tokenizer = MockTokenizer()
    
    # 计算驱动力
    print("\n开始计算驱动力 G(t)...")
    G_t_array = calculator.compute_g_t_batch(
        hidden_states=hidden_states,
        token_ids=token_ids,
        morph_features=morph_features,
        tokenizer=tokenizer
    )
    
    print(f"\n✓ 计算完成！")
    print(f"  驱动力张量形状: {G_t_array.shape}")
    print(f"  期望形状: ({seq_len}, {hidden_dim})")
    
    # 输出每个token的驱动力统计
    print(f"\n{'Token':<10} {'G(t)均值':<15} {'G(t)标准差':<15} {'G(t)范数':<15}")
    print("-" * 60)
    for i in range(seq_len):
        g_mean = np.mean(G_t_array[i])
        g_std = np.std(G_t_array[i])
        g_norm = np.linalg.norm(G_t_array[i])
        print(f"Token {i:<3} {g_mean:>12.6f}   {g_std:>12.6f}   {g_norm:>12.6f}")
    
    print("\n" + "=" * 60)
    print("测试完成！驱动力计算模块运行正常。")
    print("=" * 60)
