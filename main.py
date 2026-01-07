from llm_hidden_extractor import extract_hidden_states
from k_t_calculator import compute_local_sectional_curvature, compute_curvature_batch
from d_t_calculator import compute_d_t, compute_d_t_batch
from c_t_calculator import compute_c_t, compute_c_t_batch
from v_t_calculator import compute_v_t, compute_v_t_batch
from m_t_calculator import compute_m_t, compute_m_t_batch, init_m_t_tools
from g_t_calculator import init_driving_force_calculator
from h_t_calculator import init_entropy_calculator
from s_t_calculator import init_drift_calculator
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os

# ---------------------- 初始化辅助模型 ----------------------
def init_poly_mlp(input_dim: int = 896, num_senses: int = 5) -> nn.Module:
    mlp = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, num_senses)
    ).eval()
    return mlp

# ---------------------- 绘图函数 ----------------------
def plot_k_t_by_layer(k_t_data, token_texts, layers, text):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(layers)))
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    num_tokens = len(token_texts)
    x = np.arange(num_tokens)
    width = 0.11
    
    for i, layer in enumerate(layers):
        offset = (i - len(layers)/2) * width
        bars = ax.bar(x + offset, k_t_data[i], width, 
                     label=f'Layer {layer}', 
                     color=colors[i],
                     edgecolor='white',
                     linewidth=1.2,
                     alpha=0.85)
    
    ax.set_xlabel('Token Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sectional Curvature K(t)', fontsize=13, fontweight='bold')
    ax.set_title(f'Local Sectional Curvature Distribution\nText: "{text}"', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}\n{token}' for i, token in enumerate(token_texts)], 
                       fontsize=10)
    
    ax.legend(loc='upper right', frameon=True, fancybox=False, 
             shadow=False, framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')
    
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    filename = f'outputs/k_t_plot_{text.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nK(t)图表已保存为 {filename}")

def plot_d_t_by_layer(d_t_data, token_texts, layers, text):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(layers)))
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    num_tokens = len(token_texts)
    x = np.arange(num_tokens)
    width = 0.11
    
    for i, layer in enumerate(layers):
        offset = (i - len(layers)/2) * width
        bars = ax.bar(x + offset, d_t_data[i], width, 
                     label=f'Layer {layer}', 
                     color=colors[i],
                     edgecolor='white',
                     linewidth=1.2,
                     alpha=0.85)
    
    ax.set_xlabel('Token Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Semantic Distribution Distance D(t)', fontsize=13, fontweight='bold')
    ax.set_title(f'D(t) Distribution\nText: "{text}"', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}\n{token}' for i, token in enumerate(token_texts)], 
                       fontsize=10)
    
    ax.legend(loc='upper right', frameon=True, fancybox=False, 
             shadow=False, framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')
    
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    filename = f'outputs/d_t_plot_{text.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nD(t)图表已保存为 {filename}")

def plot_c_t_by_layer(c_t_data, token_texts, layers, text):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(layers)))
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    num_tokens = len(token_texts)
    x = np.arange(num_tokens)
    width = 0.11
    
    for i, layer in enumerate(layers):
        offset = (i - len(layers)/2) * width
        bars = ax.bar(x + offset, c_t_data[i], width, 
                     label=f'Layer {layer}', 
                     color=colors[i],
                     edgecolor='white',
                     linewidth=1.2,
                     alpha=0.85)
    
    ax.set_xlabel('Token Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Local Semantic Clustering Density C(t)', fontsize=13, fontweight='bold')
    ax.set_title(f'C(t) Distribution\nText: "{text}"', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}\n{token}' for i, token in enumerate(token_texts)], 
                       fontsize=10)
    
    ax.legend(loc='upper right', frameon=True, fancybox=False, 
             shadow=False, framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')
    
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    filename = f'outputs/c_t_plot_{text.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nC(t)图表已保存为 {filename}")

def plot_v_t_by_layer(v_t_data, token_texts, layers, text):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(layers)))
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    num_tokens = len(token_texts)
    x = np.arange(num_tokens)
    width = 0.11
    
    for i, layer in enumerate(layers):
        offset = (i - len(layers)/2) * width
        bars = ax.bar(x + offset, v_t_data[i], width, 
                     label=f'Layer {layer}', 
                     color=colors[i],
                     edgecolor='white',
                     linewidth=1.2,
                     alpha=0.85)
    
    ax.set_xlabel('Token Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Semantic Consistency Potential V(t)', fontsize=13, fontweight='bold')
    ax.set_title(f'V(t) Distribution\nText: "{text}"', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}\n{token}' for i, token in enumerate(token_texts)], 
                       fontsize=10)
    
    # 设置Y轴范围从-10开始
    ax.set_ylim(bottom=-10)
    
    ax.legend(loc='upper right', frameon=True, fancybox=False, 
             shadow=False, framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')
    
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    filename = f'outputs/v_t_plot_{text.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nV(t)图表已保存为 {filename}")

def plot_g_t_by_layer(g_t_data, token_texts, layers, text):
    """绘制上下文驱动力G(t)的范数变化（驱动力强度）"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(layers)))
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    num_tokens = len(token_texts)
    x = np.arange(num_tokens)
    width = 0.11
    
    # 计算每个token的G(t)范数（驱动力强度）
    g_t_norms = np.zeros((len(layers), num_tokens))
    for i, layer_idx in enumerate(layers):
        for j in range(num_tokens):
            g_t_norms[i, j] = np.linalg.norm(g_t_data[i, j, :])  # 896维向量的L2范数
    
    for i, layer in enumerate(layers):
        offset = (i - len(layers)/2) * width
        bars = ax.bar(x + offset, g_t_norms[i], width,
                     label=f'Layer {layer}',
                     color=colors[i],
                     edgecolor='white',
                     linewidth=1.2,
                     alpha=0.85)
    
    ax.set_xlabel('Token Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Driving Force Strength ||G(t)||', fontsize=13, fontweight='bold')
    ax.set_title(f'Context Driving Force G(t) Distribution\nText: "{text}"',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}\n{token}' for i, token in enumerate(token_texts)],
                       fontsize=10)
    
    ax.legend(loc='upper right', frameon=True, fancybox=False,
             shadow=False, framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')
    
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    filename = f'outputs/g_t_plot_{text.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nG(t)图表已保存为 {filename}")

def plot_m_t_by_layer(m_t_data, token_texts, layers, text):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(layers)))
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    num_tokens = len(token_texts)
    x = np.arange(num_tokens)
    width = 0.11
    
    for i, layer in enumerate(layers):
        offset = (i - len(layers)/2) * width
        bars = ax.bar(x + offset, m_t_data[i], width, 
                     label=f'Layer {layer}', 
                     color=colors[i],
                     edgecolor='white',
                     linewidth=1.2,
                     alpha=0.85)
    
    ax.set_xlabel('Token Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Morph-Semantic Matching Degree M(t)', fontsize=13, fontweight='bold')
    ax.set_title(f'M(t) Distribution\nText: "{text}"', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(token_texts, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    filename = f'outputs/m_t_plot_{text.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'\nM(t)图表已保存为 {filename}')

def plot_h_t_by_layer(h_t_data, token_texts, layers, text):
    """绘制多义性熵 H(t) 的多层对比图
    
    Args:
        h_t_data: [num_layers, seq_len] 每层每个token的多义性熵
        token_texts: token文本列表
        layers: 分析的层索引列表
        text: 输入文本
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(layers)))  # 红-黄-绿（反向：红=高熵）
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    num_tokens = len(token_texts)
    x = np.arange(num_tokens)
    width = 0.11
    
    for i, layer in enumerate(layers):
        offset = (i - len(layers)/2) * width
        bars = ax.bar(x + offset, h_t_data[i], width, 
                     label=f'Layer {layer}', 
                     color=colors[i],
                     edgecolor='white',
                     linewidth=1.2,
                     alpha=0.85)
    
    ax.set_xlabel('Token Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Polysemy Entropy H(t)', fontsize=13, fontweight='bold')
    ax.set_title(f'H(t) Polysemy Uncertainty Distribution\nText: "{text}"\n(Higher H(t) = More Ambiguous)', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}\n{token}' for i, token in enumerate(token_texts)], 
                       fontsize=10)
    ax.set_ylim(0, 1.0)
    
    ax.legend(loc='upper right', frameon=True, fancybox=False, 
             shadow=False, framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')
    
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    filename = f'outputs/h_t_plot_{text.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'\nH(t)图表已保存为 {filename}')
    
    plt.close()

def plot_s_t_by_layer(s_t_data, token_texts, layers, text):
    """绘制语义漂移系数 S(t) 的多层对比图
    
    Args:
        s_t_data: [num_layers, seq_len] 每层每个token的语义漂移系数
        token_texts: token文本列表
        layers: 分析的层索引列表
        text: 输入文本
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    colors = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(layers)))  # 黄-橙-红渐变（红色=高漂移）
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    num_tokens = len(token_texts)
    x = np.arange(num_tokens)
    width = 0.11
    
    for i, layer in enumerate(layers):
        offset = (i - len(layers)/2) * width
        bars = ax.bar(x + offset, s_t_data[i], width, 
                     label=f'Layer {layer}', 
                     color=colors[i],
                     edgecolor='white',
                     linewidth=1.2,
                     alpha=0.85)
    
    ax.set_xlabel('Token Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Semantic Drift Coefficient S(t)', fontsize=13, fontweight='bold')
    ax.set_title(f'S(t) Semantic Drift Distribution\nText: "{text}"\n(Higher S(t) = More Drift from Global Theme)', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}\n{token}' for i, token in enumerate(token_texts)], 
                       fontsize=10)
    ax.set_ylim(0, 1.0)
    
    ax.legend(loc='upper right', frameon=True, fancybox=False, 
             shadow=False, framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')
    
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    filename = f'outputs/s_t_plot_{text.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'\nS(t)图表已保存为 {filename}')
    
    plt.close()

# ---------------------- 主函数 ----------------------
def main():
    # 1. 配置参数
    # 对比测试：
    # "他打电话了" - "打电话"强固定搭配，"打"的多义性应该被充分消解（H(t)低）
    # "他打了一个" - "打"后缺少明确对象，多义性高（H(t)高）
    text = "他打电话了"
    layers = list(range(10, 17))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dim = 896
    morph_dim = 224  # 形态特征维度
    
    # 2. 初始化所有工具
    # 2.1 M(t)工具（真实形态-语义匹配度）
    morph_extractor, morph_embedding, m_poly_mlp = init_m_t_tools(hidden_dim=hidden_dim)
    morph_embedding = morph_embedding.to(device)
    m_poly_mlp = m_poly_mlp.to(device)
    
    # 2.2 V(t)工具
    poly_mlp = init_poly_mlp(input_dim=hidden_dim).to(device)
    
    # 2.3 G(t)工具会在第一次循环时初始化（需要tokenizer的vocab_size）
    driving_force_calc = None
    
    # 2.4 H(t)工具（多义性熵计算器）
    entropy_calc = init_entropy_calculator(hidden_dim=hidden_dim, morph_dim=morph_dim)
    
    # 2.5 S(t)工具（语义漂移系数计算器）
    drift_calc = init_drift_calculator()
    
    # 3. 存储所有指标数据
    all_k_t = []
    all_d_t = []
    all_c_t = []
    all_v_t = []
    all_m_t = []
    all_g_t = []  # 存储驱动力G(t)
    all_h_t = []  # 存储多义性熵H(t)
    all_s_t = []  # 新增：存储语义漂移系数S(t)
    token_texts = []
    
    # 4. 逐层计算
    for middle_layer in layers:
        print(f"\n{'='*60}")
        print(f"中间层编号：{middle_layer}")
        print(f"{'='*60}")
        
        # 4.1 提取真实LLM隐藏状态和注意力权重
        hidden_states, token_num, tokenizer, inputs, attn_weights = extract_hidden_states(
            text=text,
            middle_layer_idx=middle_layer,
            device=device
        )
        print(f"文本token数：{token_num}")
        
        # 4.2 提取token文本
        current_token_texts = []
        for token_idx in range(token_num):
            token_id = inputs["input_ids"][0][token_idx].item()
            token_text = tokenizer.decode([token_id])
            current_token_texts.append(token_text)
        if middle_layer == layers[0]:
            token_texts = current_token_texts
        
        # 首次循环时初始化G(t)工具（需要tokenizer的vocab_size）
        if driving_force_calc is None:
            vocab_size = len(tokenizer)  # 获取实际词表大小
            print(f"初始化驱动力计算器，vocab_size={vocab_size}")
            driving_force_calc = init_driving_force_calculator(
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                max_seq_len=512
            )
        
        # 4.3 预计算当前层真实M(t)和m(t)向量
        # M(t): 形态-语义匹配度标量（用于K/D/C计算）
        current_m_t = compute_m_t_batch(
            hidden_states=hidden_states,
            token_texts=current_token_texts,
            morph_extractor=morph_extractor,
            morph_embedding=morph_embedding,
            poly_mlp=m_poly_mlp
        )
        all_m_t.append(current_m_t)
        
        # m(t): 形态特征向量（224维，用于V_morph计算）
        from m_t_calculator import extract_m_t_batch
        current_m_t_vectors = extract_m_t_batch(
            token_texts=current_token_texts,
            morph_extractor=morph_extractor
        )
        
        # 4.4 计算K(t)（注入真实M(t)）
        current_k_t = compute_curvature_batch(
            hidden_states=hidden_states,
            sentence_length=token_num,
            window_size=3,
            sim_threshold=0.5,
            precomputed_m_t_list=current_m_t.tolist()
        )
        all_k_t.append(current_k_t)
        
        # 4.5 计算D(t)
        current_d_t = compute_d_t_batch(
            hidden_states=hidden_states,
            window_size=3,
            sim_threshold=0.5,
            epsilon=1e-6,
            precomputed_m_t_list=current_m_t
        )
        all_d_t.append(current_d_t)
        
        # 4.6 计算C(t)
        current_c_t = compute_c_t_batch(
            hidden_states=hidden_states,
            k=3,
            theta=0.5,
            alpha=0.4,
            precomputed_m_t_list=current_m_t
        )
        all_c_t.append(current_c_t)
        
        # 4.7 计算V(t) - 使用真实注意力权重和预计算的m(t)
        # 从二维注意力矩阵提取一维全局贡献度（每个token被其他所有token关注的总和）
        attn_weights_global = attn_weights.sum(dim=0)  # [seq_len] 用于V_global计算
        
        # token_groups: 每个token的高层级单元（相邻tokens，不含自身）
        token_groups = []
        for i in range(token_num):
            group = []
            if i > 0:
                group.append(i - 1)
            if i < token_num - 1:
                group.append(i + 1)
            token_groups.append(group)
        current_v_t = compute_v_t_batch(
            hidden_states=hidden_states,
            attn_weights=attn_weights_global,  # 使用一维全局贡献度
            token_groups=token_groups,
            m_t_list=current_m_t_vectors,
            morph_embedding=morph_embedding,
            poly_mlp=poly_mlp
        )
        all_v_t.append(current_v_t)
        
        # 4.8 计算G(t) - 上下文驱动力
        current_g_t = driving_force_calc.compute_g_t_batch(
            hidden_states=hidden_states,
            token_ids=inputs["input_ids"][0],
            morph_features=current_m_t_vectors,
            tokenizer=tokenizer
        )
        all_g_t.append(current_g_t)
        
        # 4.9 计算多义性熵 H(t)
        h_t_batch = entropy_calc.compute_batch_entropy(
            tokens=current_token_texts,
            hidden_states=hidden_states.squeeze(0),  # [seq_len, hidden_dim]
            morph_features=current_m_t_vectors,  # [seq_len, morph_dim]
            attention_weights=attn_weights  # 直接使用二维矩阵 [seq_len, seq_len]
        )
        all_h_t.append(h_t_batch)
        
        # 4.10 计算语义漂移系数 S(t)
        s_t_batch = drift_calc.compute_s_t_batch(
            hidden_states=hidden_states.squeeze(0),  # [seq_len, hidden_dim]
            c_t_list=current_c_t,  # [seq_len] 聚类密度
            d_t_list=current_d_t,  # [seq_len] 平均距离
            m_t_list=current_m_t   # [seq_len] 形态匹配度
        )
        all_s_t.append(s_t_batch)
        
        # 4.11 输出当前层所有指标结果
        for token_idx in range(token_num):
            print(f"Token[{token_idx}]：{current_token_texts[token_idx]}")
            print(f"  K(t) = {current_k_t[token_idx]:.6f}")
            print(f"  D(t) = {current_d_t[token_idx]:.6f}")
            print(f"  C(t) = {current_c_t[token_idx]:.6f}")
            print(f"  V(t) = {current_v_t[token_idx]:.6f}")
            print(f"  M(t) = {current_m_t[token_idx]:.6f}")
            g_norm = np.linalg.norm(current_g_t[token_idx])
            print(f"  ||G(t)|| = {g_norm:.6f}")
            print(f"  H(t) = {h_t_batch[token_idx]:.6f}")
            print(f"  S(t) = {s_t_batch[token_idx]:.6f}")
            print(f"  ------------------------")
        
        # 输出 H(t) 统计信息
        print(f"\n  Layer {middle_layer} - Polysemy Entropy H(t):")
        print(f"    Mean H(t): {h_t_batch.mean():.4f}")
        print(f"    Max H(t):  {h_t_batch.max():.4f} (Most ambiguous)")
        print(f"    Min H(t):  {h_t_batch.min():.4f} (Most certain)")
        # 找出最高熵token（最模糊）
        max_entropy_idx = h_t_batch.argmax()
        print(f"    Most ambiguous token: '{current_token_texts[max_entropy_idx]}' (H={h_t_batch[max_entropy_idx]:.4f})")
        print(f"  ------------------------")
        
        # 输出 S(t) 统计信息
        print(f"\n  Layer {middle_layer} - Semantic Drift Coefficient S(t):")
        print(f"    Mean S(t): {s_t_batch.mean():.4f}")
        print(f"    Max S(t):  {s_t_batch.max():.4f} (Most drift)")
        print(f"    Min S(t):  {s_t_batch.min():.4f} (Most stable)")
        # 找出最高漂移token
        max_drift_idx = s_t_batch.argmax()
        print(f"    Most drifted token: '{current_token_texts[max_drift_idx]}' (S={s_t_batch[max_drift_idx]:.4f})")
        print(f"  ------------------------")
    
    # 5. 转换为数组并绘图
    k_t_array = np.array(all_k_t)
    d_t_array = np.array(all_d_t)
    c_t_array = np.array(all_c_t)
    v_t_array = np.array(all_v_t)
    m_t_array = np.array(all_m_t)
    g_t_array = np.array(all_g_t)  # (num_layers, seq_len, hidden_dim)
    h_t_array = np.array(all_h_t)  # (num_layers, seq_len)
    s_t_array = np.array(all_s_t)  # (num_layers, seq_len)
    
    plot_k_t_by_layer(k_t_array, token_texts, layers, text)
    plot_d_t_by_layer(d_t_array, token_texts, layers, text)
    plot_c_t_by_layer(c_t_array, token_texts, layers, text)
    plot_v_t_by_layer(v_t_array, token_texts, layers, text)
    plot_m_t_by_layer(m_t_array, token_texts, layers, text)
    plot_g_t_by_layer(g_t_array, token_texts, layers, text)
    plot_h_t_by_layer(h_t_array, token_texts, layers, text)
    plot_s_t_by_layer(s_t_array, token_texts, layers, text)

if __name__ == "__main__":
    main()