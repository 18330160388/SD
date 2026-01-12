from llm_hidden_extractor import extract_hidden_states
from k_t_calculator import compute_local_sectional_curvature, compute_curvature_batch
from d_t_calculator import compute_d_t, compute_d_t_batch
from c_t_calculator import compute_c_t, compute_c_t_batch
from v_t_calculator import compute_v_t, compute_v_t_batch
from m_t_calculator import compute_m_t, compute_m_t_batch, init_m_t_tools
from g_t_calculator import init_driving_force_calculator
from h_t_calculator import init_entropy_calculator
from s_t_calculator import init_drift_calculator
from flow_rate_calculator import init_flow_calculator
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
    
    # 自动调整Y轴范围以显示负值
    data_min = np.min(d_t_data)
    data_max = np.max(d_t_data)
    data_range = data_max - data_min
    margin = 0.15 * data_range if data_range > 1e-6 else 0.1
    ax.set_ylim(data_min - margin, data_max + margin)
    
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
    
    # 自动调整Y轴范围以显示负值
    data_min = np.min(c_t_data)
    data_max = np.max(c_t_data)
    data_range = data_max - data_min
    margin = 0.15 * data_range if data_range > 1e-6 else 0.1
    ax.set_ylim(data_min - margin, data_max + margin)
    
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
        
        # 在每个柱子上方标注数值
        for j, bar in enumerate(bars):
            height = bar.get_height()
            # 根据数值大小调整标注位置和格式
            if abs(height) >= 10:
                label_text = f'{height:.1f}'
                fontsize = 7
            elif abs(height) >= 1:
                label_text = f'{height:.2f}'
                fontsize = 7
            else:
                label_text = f'{height:.3f}'
                fontsize = 6
            
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label_text,
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=fontsize, rotation=0,
                   color='black', fontweight='normal')
    
    ax.set_xlabel('Token Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Semantic Consistency Potential V(t)', fontsize=13, fontweight='bold')
    ax.set_title(f'V(t) Distribution\nText: "{text}"', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}\n{token}' for i, token in enumerate(token_texts)], 
                       fontsize=10)
    
    # 自动调整Y轴范围以显示负值
    data_min = np.min(v_t_data)
    data_max = np.max(v_t_data)
    data_range = data_max - data_min
    margin = 0.15 * data_range if data_range > 1e-6 else 0.1
    ax.set_ylim(data_min - margin, data_max + margin)
    
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
    
    # 自动调整Y轴范围以显示负值
    data_min = np.min(g_t_norms)
    data_max = np.max(g_t_norms)
    data_range = data_max - data_min
    margin = 0.15 * data_range if data_range > 1e-6 else 0.1
    ax.set_ylim(data_min - margin, data_max + margin)
    
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
    
    # 自动调整Y轴范围以显示负值
    data_min = np.min(m_t_data)
    data_max = np.max(m_t_data)
    data_range = data_max - data_min
    margin = 0.15 * data_range if data_range > 1e-6 else 0.1
    ax.set_ylim(data_min - margin, data_max + margin)
    
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
    
    # 自动调整Y轴范围以显示负值
    data_min = np.min(h_t_data)
    data_max = np.max(h_t_data)
    data_range = data_max - data_min
    margin = 0.15 * data_range if data_range > 1e-6 else 0.1
    ax.set_ylim(data_min - margin, data_max + margin)
    
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
    
    # 自动调整Y轴范围以显示负值
    data_min = np.min(s_t_data)
    data_max = np.max(s_t_data)
    data_range = data_max - data_min
    margin = 0.15 * data_range if data_range > 1e-6 else 0.1
    ax.set_ylim(data_min - margin, data_max + margin)
    
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

# ---------------------- 流率和驱动力绘图函数 ----------------------
def plot_flow_rates_by_layer(flow_data, token_texts, layers, text, metric_name, ylabel, colormap='coolwarm'):
    """
    绘制流率指标的跨层变化图（折线图）
    每个token转换绘制一条曲线，展示其在不同层之间的演化趋势
    
    参数:
        flow_data: (num_layers, seq_len-1) 流率数据（注意长度比token少1）
        token_texts: token文本列表
        layers: 层编号列表
        text: 原始文本
        metric_name: 指标名称（如'dK_dt'）
        ylabel: Y轴标签
        colormap: 颜色映射
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    # 注意：流率长度比token少1（因为是差分）
    num_flows = flow_data.shape[1]
    
    # 为每个token转换生成颜色
    import matplotlib
    colors = matplotlib.colormaps.get_cmap(colormap)(np.linspace(0.2, 0.8, num_flows))
    
    # 绘制每个token转换的演化曲线
    for j in range(num_flows):
        transition_label = f'{token_texts[j]}→{token_texts[j+1] if j+1 < len(token_texts) else "END"}'
        ax.plot(layers, flow_data[:, j], 
               marker='o', markersize=8, linewidth=2.5,
               label=transition_label, color=colors[j],
               alpha=0.85)
    
    # 添加零线（参考线）
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
    
    ax.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_title(f'{ylabel}\nText: "{text}"', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xticks(layers)
    ax.set_xticklabels([f'L{l}' for l in layers], fontsize=10)
    
    # 自动调整Y轴范围以显示负值
    data_min = np.min(flow_data)
    data_max = np.max(flow_data)
    data_range = data_max - data_min
    margin = 0.15 * data_range if data_range > 1e-6 else 0.1
    ax.set_ylim(data_min - margin, data_max + margin)
    
    # 图例
    ax.legend(loc='best', frameon=True, fancybox=True,
             shadow=True, framealpha=0.9, fontsize=9)
    
    # 网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # 边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')
    
    plt.tight_layout()
    filename = f'outputs/{metric_name}_plot_{text.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'[{metric_name}图表] 已保存到 {filename}')
    plt.close()

# ---------------------- 主函数 ----------------------
def main():
    # 1. 配置参数
    # 对比测试：
    # "他打电话了" - "打电话"强固定搭配，"打"的多义性应该被充分消解（H(t)低）
    # "他打了一个" - "打"后缺少明确对象，多义性高（H(t)高）
    # text = "书和笔它很轻便"
    text = "他很行"
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
    
    # 2.6 流率计算器
    flow_calc = init_flow_calculator()
    
    # 3. 存储所有指标数据
    all_k_t = []
    all_d_t = []
    all_c_t = []
    all_v_t = []
    all_m_t = []
    all_g_t = []  # 存储驱动力G(t)
    all_h_t = []  # 存储多义性熵H(t)
    all_s_t = []  # 新增：存储语义漂移系数S(t)
    all_hidden_states = []  # 存储隐状态（用于计算流率）
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
        
        # 4.11 存储隐状态（用于后续流率计算）
        all_hidden_states.append(hidden_states.squeeze(0).cpu().numpy())
        
        # 4.12 单token输出
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
    hidden_states_array = np.array(all_hidden_states)  # (num_layers, seq_len, hidden_dim)
    
    # 5.1 计算流率指标
    print(f"\n{'='*60}")
    print("计算流率指标（系统动力学流量）")
    print(f"{'='*60}")
    
    all_dK_dt = []
    all_dD_dt = []
    all_dC_dt = []
    all_dV_dt = []
    
    for layer_idx in range(len(layers)):
        # 提取当前层的状态序列
        k_seq = k_t_array[layer_idx].tolist()
        d_seq = d_t_array[layer_idx].tolist()
        c_seq = c_t_array[layer_idx].tolist()
        v_seq = v_t_array[layer_idx].tolist()
        h_states = torch.from_numpy(hidden_states_array[layer_idx]).float()
        
        # 计算流率
        flow_rates = flow_calc.compute_flow_rates(
            k_sequence=k_seq,
            d_sequence=d_seq,
            c_sequence=c_seq,
            v_sequence=v_seq,
            hidden_states=h_states
        )
        
        all_dK_dt.append(flow_rates['dK_dt'])
        all_dD_dt.append(flow_rates['dD_dt'])
        all_dC_dt.append(flow_rates['dC_dt'])
        all_dV_dt.append(flow_rates['dV_dt'])
        
        # 输出当前层的流率统计
        layer_num = layers[layer_idx]
        print(f"\n--- Layer {layer_num} 流率统计 ---")
        stats = flow_calc.compute_flow_statistics(flow_rates)
        for name, stat in stats.items():
            print(f"{name}: mean={stat['mean']:.4f}, max={stat['max']:.4f}, min={stat['min']:.4f}")
        
        # 输出dV/dt与-G(t)的相关性
        if 'dV_G_correlation' in flow_rates:
            print(f"dV/dt与-G(t)相关性: {flow_rates['dV_G_correlation']:.4f}")
        
        # 识别临界点
        critical_points = flow_calc.identify_critical_points(flow_rates, threshold_percentile=90.0)
        print(f"临界点（top 10%变化最剧烈的位置）:")
        for name, indices in critical_points.items():
            if len(indices) > 0:
                transitions = [f"{i}→{i+1}" for i in indices]
                print(f"  {name}: {', '.join(transitions)}")
    
    # 转换为数组
    dK_dt_array = np.array(all_dK_dt)  # (num_layers, seq_len-1)
    dD_dt_array = np.array(all_dD_dt)
    dC_dt_array = np.array(all_dC_dt)
    dV_dt_array = np.array(all_dV_dt)
    
    print(f"\n{'='*60}")
    print("绘制所有图表")
    print(f"{'='*60}")
    
    # 5.2 绘制存量指标（原有的8个指标）
    plot_k_t_by_layer(k_t_array, token_texts, layers, text)
    plot_d_t_by_layer(d_t_array, token_texts, layers, text)
    plot_c_t_by_layer(c_t_array, token_texts, layers, text)
    plot_v_t_by_layer(v_t_array, token_texts, layers, text)
    plot_m_t_by_layer(m_t_array, token_texts, layers, text)
    plot_g_t_by_layer(g_t_array, token_texts, layers, text)
    plot_h_t_by_layer(h_t_array, token_texts, layers, text)
    plot_s_t_by_layer(s_t_array, token_texts, layers, text)
    
    # 5.3 绘制流率指标（新增的4个流量指标）
    plot_flow_rates_by_layer(dK_dt_array, token_texts, layers, text, 
                            'dK_dt', 'Curvature Evolution Rate dK/dt')
    plot_flow_rates_by_layer(dD_dt_array, token_texts, layers, text,
                            'dD_dt', 'Distance Change Rate dD/dt')
    plot_flow_rates_by_layer(dC_dt_array, token_texts, layers, text,
                            'dC_dt', 'Clustering Density Growth Rate dC/dt')
    plot_flow_rates_by_layer(dV_dt_array, token_texts, layers, text,
                            'dV_dt', 'Potential Decay Rate dV/dt')
    
    print(f"\n{'='*60}")
    print("所有计算和绘图完成！")
    print(f"共生成12个图表：8个存量指标 + 4个流率指标")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()