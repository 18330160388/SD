from llm_hidden_extractor import extract_hidden_states
from k_t_calculator import compute_local_sectional_curvature, compute_curvature_batch
from d_t_calculator import compute_d_t, compute_d_t_batch
from c_t_calculator import compute_c_t, compute_c_t_batch
from v_t_calculator import compute_v_t, compute_v_t_batch
from m_t_calculator import compute_m_t, compute_m_t_batch, init_m_t_tools
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
    filename = f'outputs/m_t_plot_{text.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nM(t)图表已保存为 {filename}")

# ---------------------- 主函数 ----------------------
def main():
    # 1. 配置参数
    text = "他打补丁"
    layers = list(range(10, 17))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dim = 896
    
    # 2. 初始化所有工具
    # 2.1 M(t)工具（真实形态-语义匹配度）
    morph_extractor, morph_embedding, m_poly_mlp = init_m_t_tools(hidden_dim=hidden_dim)
    morph_embedding = morph_embedding.to(device)
    m_poly_mlp = m_poly_mlp.to(device)
    
    # 2.2 V(t)工具
    poly_mlp = init_poly_mlp(input_dim=hidden_dim).to(device)
    
    # 3. 存储所有指标数据
    all_k_t = []
    all_d_t = []
    all_c_t = []
    all_v_t = []
    all_m_t = []
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
            attn_weights=attn_weights,
            token_groups=token_groups,
            m_t_list=current_m_t_vectors,
            morph_embedding=morph_embedding,
            poly_mlp=poly_mlp
        )
        all_v_t.append(current_v_t)
        
        # 4.8 输出当前层结果
        for token_idx in range(token_num):
            print(f"Token[{token_idx}]：{current_token_texts[token_idx]}")
            print(f"  K(t) = {current_k_t[token_idx]:.6f}")
            print(f"  D(t) = {current_d_t[token_idx]:.6f}")
            print(f"  C(t) = {current_c_t[token_idx]:.6f}")
            print(f"  V(t) = {current_v_t[token_idx]:.6f}")
            print(f"  M(t) = {current_m_t[token_idx]:.6f}")
            print(f"  ------------------------")
    
    # 5. 转换为数组并绘图
    k_t_array = np.array(all_k_t)
    d_t_array = np.array(all_d_t)
    c_t_array = np.array(all_c_t)
    v_t_array = np.array(all_v_t)
    m_t_array = np.array(all_m_t)
    
    plot_k_t_by_layer(k_t_array, token_texts, layers, text)
    plot_d_t_by_layer(d_t_array, token_texts, layers, text)
    plot_c_t_by_layer(c_t_array, token_texts, layers, text)
    plot_v_t_by_layer(v_t_array, token_texts, layers, text)
    plot_m_t_by_layer(m_t_array, token_texts, layers, text)

if __name__ == "__main__":
    main()