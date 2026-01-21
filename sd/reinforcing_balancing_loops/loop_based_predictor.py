# 系统动力学增强-调节回路模型：语义演化预测
# 基于经典系统动力学理论：增强回路 + 调节回路 + 时滞反馈
# 严格遵循回路因果关系和反馈机制

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from scipy.optimize import minimize

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 尝试设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
except:
    pass

# 强制使用支持中文的字体
import matplotlib.font_manager as fm
try:
    # 查找系统中文字体
    font_path = None
    for font in fm.findSystemFonts():
        if 'simhei' in font.lower() or 'microsoft yahei' in font.lower():
            font_path = font
            break

    if font_path:
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
    else:
        # 如果找不到中文字体，使用默认设置
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']

# ====================== 系统动力学回路理论基础 ======================
"""
系统动力学增强-调节回路模型

核心回路结构：

1. 增强回路（Reinforcing Loop R1）：
   语义聚合速率 →[+] 曲率 →[+] 下一层聚合速率
   (前层聚合驱动后层聚合，形成正反馈)

2. 调节回路（Balancing Loop B1）：
   曲率 →[+] 语义稳定性 →[-] 聚合速率
   (聚合过度时通过稳定性机制抑制)

3. 调节回路（Balancing Loop B2）：
   曲率 →[+] 距离变化 →[+] 聚合阻力 →[-] 聚合速率
   (聚合过度时通过距离变化增加阻力)

4. 时滞反馈回路（Delayed Feedback）：
   当前聚合 →[时滞] 影响未来聚合

数学表达：
曲率演化速率(L) = R1_gain × 曲率演化速率(L-1) × (1 - B1_effect - B2_effect)

其中：
- R1_gain: 增强回路增益（正反馈强度）
- B1_effect: 调节回路B1效果（语义稳定性抑制）
- B2_effect: 调节回路B2效果（距离变化抑制）
"""

# ====================== 回路参数设置 ======================
# 增强回路参数
R1_GAIN = 0.6          # 增强回路增益 (0.4-0.8)

# 调节回路B1参数（语义稳定性）
B1_SEMANTIC_GAIN = 0.3 # 语义稳定性增益
B1_SEMANTIC_DELAY = 1  # 语义反馈延迟

# 调节回路B2参数（距离变化）
B2_DISTANCE_GAIN = 0.2 # 距离变化增益
B2_DISTANCE_THRESHOLD = 0.001  # 距离变化阈值

# 时滞参数
FEEDBACK_DELAY = 1     # 整体反馈延迟层数

# ====================== 回路状态变量 ======================
class LoopState:
    """系统动力学回路状态"""
    def __init__(self):
        self.semantic_stability = 0.0  # 语义稳定性状态
        self.distance_resistance = 0.0 # 距离阻力状态
        self.aggregation_momentum = 0.0 # 聚合动量

# ====================== 增强回路计算函数 ======================
def reinforcing_loop_effect(prev_dKdt, current_state, r1_gain=R1_GAIN):
    """
    计算增强回路效果：前层聚合驱动当前层聚合

    参数：
    prev_dKdt: 前层聚合速率
    current_state: 当前回路状态
    r1_gain: 增强回路增益

    返回：
    增强回路贡献值
    """
    # 增强回路：前层聚合正向推动当前层聚合
    reinforcing_effect = r1_gain * prev_dKdt

    # 加入动量效应（增强回路积累）
    current_state.aggregation_momentum = 0.8 * current_state.aggregation_momentum + 0.2 * reinforcing_effect

    return current_state.aggregation_momentum

# ====================== 调节回路B1计算函数 ======================
def balancing_loop_b1_effect(current_K, current_S, current_state=None,
                           b1_gain=B1_SEMANTIC_GAIN, b1_delay=B1_SEMANTIC_DELAY):
    """
    计算调节回路B1效果：语义稳定性抑制过度聚合

    参数：
    current_K: 当前曲率（实际观测值）
    current_S: 当前语义漂移系数（实际观测值）
    current_state: 当前回路状态（可选，用于兼容性）

    返回：
    调节回路B1抑制效果（完全基于实际观测值）
    """
    # 语义稳定性 = 1 / (1 + 语义漂移²) - 直接基于当前实际观测值
    semantic_stability = 1.0 / (1.0 + current_S ** 2)

    # 调节回路：根据聚合方向决定抑制强度
    # 如果K为正（聚合），稳定性抑制聚合；如果K为负（离散），稳定性抑制离散
    if current_K >= 0:
        # 正聚合时，稳定性抑制过度聚合
        stability_suppression = b1_gain * semantic_stability * current_K
    else:
        # 负聚合时，稳定性抑制过度离散（产生正向调节）
        stability_suppression = -b1_gain * semantic_stability * abs(current_K)

    return stability_suppression

# ====================== 调节回路B2计算函数 ======================
def balancing_loop_b2_effect(current_D, prev_D, current_state=None, current_K=None,
                           b2_gain=B2_DISTANCE_GAIN, b2_threshold=B2_DISTANCE_THRESHOLD):
    """
    计算调节回路B2效果：距离变化增加聚合阻力

    参数：
    current_D: 当前距离（实际观测值）
    prev_D: 前层距离（实际观测值）
    current_state: 当前回路状态（可选，用于兼容性）
    current_K: 当前曲率（实际观测值，用于判断趋势方向）

    返回：
    调节回路B2抑制效果（完全基于实际观测值）
    """
    # 距离变化幅度和方向 - 直接基于实际观测值
    distance_change = current_D - prev_D  # 保持符号

    # 距离阻力：根据聚合方向决定阻力
    if abs(distance_change) > b2_threshold:
        # 基础阻力
        base_resistance = b2_gain * (abs(distance_change) / b2_threshold)

        # 根据聚合方向调整阻力方向
        if current_K is not None:
            if current_K >= 0:
                # 正聚合时，距离增加产生阻力，距离减少减少阻力
                distance_resistance = base_resistance * (distance_change / abs(distance_change)) if distance_change != 0 else 0
            else:
                # 负聚合时，距离增加可能不是阻力（离散趋势），所以减小阻力
                distance_resistance = base_resistance * 0.5 * (distance_change / abs(distance_change)) if distance_change != 0 else 0
        else:
            # 如果没有K信息，使用传统方法（总是正阻力）
            distance_resistance = base_resistance
    else:
        distance_resistance = 0.0

    return distance_resistance

# ====================== 主预测函数 ======================
def loop_based_predict(token_data, start_layer=12, end_layer=16):
    """
    基于增强-调节回路组合的系统动力学预测

    参数：
    token_data: 单个token的层数据
    start_layer: 预测开始层
    end_layer: 预测结束层

    返回：
    预测结果字典
    """
    # 提取数据
    layers = token_data['层'].values
    K_actual = token_data['曲率 K(t)'].values
    D_actual = token_data['平均欧氏距离 D(t)'].values
    S_actual = token_data['语义漂移系数 S(t)'].values
    dKdt_actual = token_data['曲率演化速率'].values

    # 初始化回路状态
    loop_state = LoopState()

    # 初始化预测数组
    predicted_K = []
    predicted_dKdt = []

    # 根据start_layer计算初始化层（预测start_layer需要start_layer-1层的数据）
    init_layer = start_layer - 1
    init_layer_idx = np.where(layers == init_layer)[0]
    if len(init_layer_idx) == 0:
        return None

    idx_init = init_layer_idx[0]
    current_K = K_actual[idx_init]  # 从初始化层的实际K开始
    current_D = D_actual[idx_init]
    current_S = S_actual[idx_init]

    # 计算初始前层聚合速率
    prev_dKdt = dKdt_actual[idx_init]  # 初始化层的实际曲率演化速率

    # 预测每一层
    for layer in range(start_layer, end_layer + 1):
        # 获取前一层实际观测数据
        prev_layer = layer - 1
        prev_idx = np.where(layers == prev_layer)[0]
        print(f"预测层: {layer}, 前层: {prev_layer}, prev_idx: {prev_idx}, 当前K: {current_K}, 当前D: {current_D}, 当前S: {current_S}")
        if len(prev_idx) > 0:
            # 使用前一层的实际观测数据
            current_K = K_actual[prev_idx[0]]  # 预测起点是前一层的实际K
            S_prev = S_actual[prev_idx[0]]
            D_prev = D_actual[prev_idx[0]]
            prev_dKdt = dKdt_actual[prev_idx[0]]  # 前一层的实际曲率演化速率
            print(f"  用前层实际观测值: K_prev={current_K}, S_prev={S_prev}, D_prev={D_prev}, prev_dKdt={prev_dKdt}")
        else:
            # 如果前一层没有实际数据，使用预测数据（理论上不应该发生）
            S_prev = current_S
            D_prev = current_D
            print(f"  用预测值 S_prev={S_prev}, D_prev={D_prev}")

        # 计算各个回路的效果 - 完全基于实际观测值
        r1_effect = reinforcing_loop_effect(prev_dKdt, loop_state)
        b1_effect = balancing_loop_b1_effect(current_K, S_prev)
        
        # 获取前前层距离用于B2回路
        prev_prev_layer = layer - 2
        prev_prev_idx = np.where(layers == prev_prev_layer)[0]
        prev_prev_D = D_actual[prev_prev_idx[0]] if len(prev_prev_idx) > 0 else D_prev
        
        b2_effect = balancing_loop_b2_effect(D_prev, prev_prev_D, current_K=current_K)

        # 综合回路效果：基础趋势 + 调节修正
        # 基础趋势：保持前一层变化趋势
        base_trend = prev_dKdt
        # 调节修正：回路效果提供微调
        regulation_factor = 1.0 + r1_effect * 0.1 - b1_effect * 0.1 - b2_effect * 0.1
        
        net_flow_rate = base_trend * regulation_factor

        # 预测当前层曲率
        predicted_K_val = current_K + net_flow_rate
        predicted_dKdt_val = net_flow_rate

        # 存储预测结果
        predicted_K.append(predicted_K_val)
        predicted_dKdt.append(predicted_dKdt_val)

        # 更新当前状态为预测值（用于下一轮预测，如果需要的话）
        current_K = predicted_K_val
        current_D = D_prev if len(prev_idx) > 0 else current_D
        current_S = S_prev if len(prev_idx) > 0 else current_S

    return {
        'layers': list(range(start_layer, end_layer + 1)),
        'predicted_K': predicted_K,
        'predicted_dKdt': predicted_dKdt,
        'loop_states': {
            'semantic_stability': loop_state.semantic_stability,
            'distance_resistance': loop_state.distance_resistance,
            'aggregation_momentum': loop_state.aggregation_momentum
        }
    }

# ====================== 回路分析函数 ======================
def analyze_loop_dynamics(token_data, prediction_result):
    """
    分析回路动态行为
    """
    layers = prediction_result['layers']
    loop_states = prediction_result['loop_states']

    print("回路动态分析:")
    print(f"  最终语义稳定性: {loop_states['semantic_stability']:.4f}")
    print(f"  最终距离阻力: {loop_states['distance_resistance']:.4f}")
    print(f"  最终聚合动量: {loop_states['aggregation_momentum']:.4f}")

    # 分析主导回路
    if abs(loop_states['aggregation_momentum']) > abs(loop_states['semantic_stability'] + loop_states['distance_resistance']):
        print("  主导回路: 增强回路 (正反馈主导)")
    else:
        print("  主导回路: 调节回路 (负反馈主导)")

# ====================== 可视化函数 ======================
def plot_loop_comparison(actual_data, predicted_data, token_name, save_path):
    """
    生成回路模型预测vs实际对比图
    """
    layers = predicted_data['layers']
    pred_K = predicted_data['predicted_K']
    pred_dKdt = predicted_data['predicted_dKdt']

    # 提取实际数据
    actual_K = []
    actual_dKdt = []
    for layer in layers:
        actual_row = actual_data[actual_data['层'] == layer]
        if len(actual_row) > 0:
            actual_K.append(actual_row['曲率 K(t)'].values[0])
            actual_dKdt.append(actual_row['曲率演化速率'].values[0])
        else:
            actual_K.append(np.nan)
            actual_dKdt.append(np.nan)

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 曲率对比
    ax1.plot(layers, actual_K, 'b-o', label='实际曲率 K(t)', linewidth=2, markersize=6)
    ax1.plot(layers, pred_K, 'r--s', label='预测曲率 K(t)', linewidth=2, markersize=6)
    ax1.set_title(f'系统动力学增强-调节回路模型：{token_name}语义演化预测', fontsize=14, fontweight='bold')
    ax1.set_xlabel('层数', fontsize=12)
    ax1.set_ylabel('曲率 K(t)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 添加回路说明
    ax1.text(0.02, 0.98, 'R1: 增强回路\nB1: 语义稳定性调节\nB2: 距离变化调节',
             transform=ax1.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 曲率变化率对比
    ax2.plot(layers, actual_dKdt, 'b-o', label='实际曲率演化速率', linewidth=2, markersize=6)
    ax2.plot(layers, pred_dKdt, 'r--s', label='预测曲率演化速率', linewidth=2, markersize=6)
    ax2.set_xlabel('层数', fontsize=12)
    ax2.set_ylabel('曲率演化速率', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    print(f"回路模型对比图已保存到: {save_path}")

def plot_all_tokens_comparison(all_results, save_path):
    """
    生成所有token的综合对比图 - 现代化美化版本
    """
    n_tokens = len(all_results)
    if n_tokens == 0:
        return

    # 为6个token设计2x3布局
    if n_tokens == 6:
        rows, cols = 2, 3
    else:
        cols = int(np.ceil(np.sqrt(n_tokens)))
        rows = int(np.ceil(n_tokens / cols))

    # 计算全局Y轴范围（基于实际观测值）
    global_k_min, global_k_max = float('inf'), float('-inf')
    global_dkdt_min, global_dkdt_max = float('inf'), float('-inf')
    
    for result in all_results:
        actual_data = result['actual_data']
        prediction = result['prediction']
        layers = prediction['layers']
        
        # 收集该token的实际数据
        actual_K = []
        actual_dKdt = []
        for layer in layers:
            actual_row = actual_data[actual_data['层'] == layer]
            if len(actual_row) > 0:
                actual_K.append(actual_row['曲率 K(t)'].values[0])
                actual_dKdt.append(actual_row['曲率演化速率'].values[0])
        
        if actual_K:
            global_k_min = min(global_k_min, min(actual_K))
            global_k_max = max(global_k_max, max(actual_K))
        if actual_dKdt:
            global_dkdt_min = min(global_dkdt_min, min(actual_dKdt))
            global_dkdt_max = max(global_dkdt_max, max(actual_dKdt))
    
    # 为Y轴添加一些边距
    k_margin = (global_k_max - global_k_min) * 0.1
    dkdt_margin = (global_dkdt_max - global_dkdt_min) * 0.1
    
    global_k_min -= k_margin
    global_k_max += k_margin
    global_dkdt_min -= dkdt_margin
    global_dkdt_max += dkdt_margin

    # 创建大图 - 更大的尺寸和更高的分辨率
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), dpi=200,
                            facecolor='#f8f9fa', constrained_layout=True)

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # 美化主标题 - 使用现代字体和颜色
    fig.suptitle('系统动力学语义演化预测模型\n(Transformer 12-16层)', fontsize=20,
                fontweight='bold', y=0.98, color='#1f2937')

    # 现代配色方案
    colors = {
        'actual': '#3b82f6',      # 蓝色 - 实际值
        'predicted': '#ef4444',  # 红色 - 预测值
        'grid': '#e5e7eb',       # 浅灰色网格
        'background': '#ffffff', # 白色背景
        'text': '#374151'        # 深灰色文字
    }

    for idx, result in enumerate(all_results):
        row = idx // cols
        col = idx % cols

        token = result['token']
        prediction = result['prediction']
        actual_data = result['actual_data']
        metrics = result['metrics']

        layers = prediction['layers']
        pred_K = prediction['predicted_K']
        pred_dKdt = prediction['predicted_dKdt']

        # 提取实际数据
        actual_K = []
        actual_dKdt = []
        for layer in layers:
            actual_row = actual_data[actual_data['层'] == layer]
            if len(actual_row) > 0:
                actual_K.append(actual_row['曲率 K(t)'].values[0])
                actual_dKdt.append(actual_row['曲率演化速率'].values[0])
            else:
                actual_K.append(np.nan)
                actual_dKdt.append(np.nan)

        ax = axes[row, col]

        # 设置子图背景
        ax.set_facecolor(colors['background'])
        ax.patch.set_alpha(0.8)

        # 创建两个子图：左边曲率，右边变化率
        # 左半部分：曲率 K(t)
        ax_left = ax.inset_axes([0.05, 0.15, 0.4, 0.75])
        ax_left.plot(layers, actual_K, 'o-', color=colors['actual'],
                    label='实际观测', linewidth=2.5, markersize=8,
                    markerfacecolor='white', markeredgewidth=2.5, markeredgecolor=colors['actual'])
        ax_left.plot(layers, pred_K, 's--', color=colors['predicted'],
                    label='动力学预测', linewidth=2.5, markersize=8,
                    markerfacecolor='white', markeredgewidth=2.5, markeredgecolor=colors['predicted'])

        # 设置固定的Y轴范围（基于全局实际观测值范围）
        ax_left.set_ylim(global_k_min, global_k_max)

        ax_left.set_title(f'{token} 语义曲率', fontsize=13, fontweight='bold',
                         color=colors['text'], pad=15)
        ax_left.set_xlabel('层级', fontsize=11, color=colors['text'])
        ax_left.set_ylabel('K(t) 值', fontsize=11, color=colors['text'])
        ax_left.legend(fontsize=9, frameon=True, fancybox=True, framealpha=0.9,
                      shadow=True, loc='best')
        ax_left.grid(True, alpha=0.4, linestyle='--', color=colors['grid'], linewidth=0.8)
        ax_left.tick_params(axis='both', which='major', labelsize=10, colors=colors['text'])
        ax_left.spines['top'].set_visible(False)
        ax_left.spines['right'].set_visible(False)
        ax_left.spines['left'].set_linewidth(0.8)
        ax_left.spines['bottom'].set_linewidth(0.8)

        # 右半部分：曲率演化速率
        ax_right = ax.inset_axes([0.55, 0.15, 0.4, 0.75])
        ax_right.plot(layers, actual_dKdt, 'o-', color=colors['actual'],
                     label='实际观测', linewidth=2.5, markersize=8,
                     markerfacecolor='white', markeredgewidth=2.5, markeredgecolor=colors['actual'])
        ax_right.plot(layers, pred_dKdt, 's--', color=colors['predicted'],
                     label='动力学预测', linewidth=2.5, markersize=8,
                     markerfacecolor='white', markeredgewidth=2.5, markeredgecolor=colors['predicted'])

        # 设置固定的Y轴范围（基于全局实际观测值范围）
        ax_right.set_ylim(global_dkdt_min, global_dkdt_max)

        ax_right.set_title(f'{token} 变化率', fontsize=13, fontweight='bold',
                          color=colors['text'], pad=15)
        ax_right.set_xlabel('层级', fontsize=11, color=colors['text'])
        ax_right.set_ylabel('曲率演化速率', fontsize=11, color=colors['text'])
        ax_right.legend(fontsize=9, frameon=True, fancybox=True, framealpha=0.9,
                       shadow=True, loc='best')
        ax_right.grid(True, alpha=0.4, linestyle='--', color=colors['grid'], linewidth=0.8)
        ax_right.tick_params(axis='both', which='major', labelsize=10, colors=colors['text'])
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['left'].set_linewidth(0.8)
        ax_right.spines['bottom'].set_linewidth(0.8)

        # 在底部添加MAE信息 - 现代化设计
        mae_value = metrics['MAE']
        if mae_value < 0.0005:
            mae_color = '#10b981'  # 绿色 - 优秀
            mae_status = '优秀'
        elif mae_value < 0.001:
            mae_color = '#f59e0b'  # 橙色 - 良好
            mae_status = '良好'
        else:
            mae_color = '#ef4444'  # 红色 - 需要改进
            mae_status = '需改进'

        # 添加MAE信息框
        ax.text(0.5, 0.02, f'{mae_status} MAE: {mae_value:.6f}',
                ha='center', va='bottom', transform=ax.transAxes,
                fontsize=12, fontweight='bold', color=mae_color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor=mae_color, linewidth=2, alpha=0.95))

        # 隐藏主轴
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # 添加子图边框
        for spine in ax.spines.values():
            spine.set_visible(False)

    # 隐藏多余的子图
    for idx in range(n_tokens, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)

    # 添加底部说明
    fig.text(0.5, 0.02,
            '基于系统动力学增强-调节回路模型 | 蓝色圆圈：实际观测 | 红色方块：动力学预测',
            ha='center', fontsize=11, color='#6b7280',
            style='italic')

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa', pad_inches=0.5)
    plt.close()

    print(f"🎨 现代化美化综合对比图已保存到: {save_path}")
    n_tokens = len(all_results)
    if n_tokens == 0:
        return

    # 计算子图布局 (尽量接近正方形)
    cols = int(np.ceil(np.sqrt(n_tokens)))
    rows = int(np.ceil(n_tokens / cols))

    # 创建大图 - 更大的尺寸和更高的分辨率
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), dpi=150)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # 美化主标题
    fig.suptitle('系统动力学增强-调节回路模型：句子语义演化预测\n(12-16层)', fontsize=18, fontweight='bold', y=0.98, color='#2E3440')

    for idx, result in enumerate(all_results):
        row = idx // cols
        col = idx % cols

        token = result['token']
        prediction = result['prediction']
        actual_data = result['actual_data']
        metrics = result['metrics']

        layers = prediction['layers']
        pred_K = prediction['predicted_K']
        pred_dKdt = prediction['predicted_dKdt']

        # 提取实际数据
        actual_K = []
        actual_dKdt = []
        for layer in layers:
            actual_row = actual_data[actual_data['层'] == layer]
            if len(actual_row) > 0:
                actual_K.append(actual_row['曲率 K(t)'].values[0])
                actual_dKdt.append(actual_row['曲率演化速率'].values[0])
            else:
                actual_K.append(np.nan)
                actual_dKdt.append(np.nan)

        ax = axes[row, col]

        # 在同一子图中绘制两个图表（上下排列）- 美化版本
        # 上半部分：曲率 - 使用更美观的样式
        ax_top = ax.inset_axes([0.08, 0.55, 0.88, 0.4])  # 调整边距
        ax_top.plot(layers, actual_K, 'o-', color='#5E81AC', label='实际观测', linewidth=2, markersize=6, markerfacecolor='white', markeredgewidth=2)
        ax_top.plot(layers, pred_K, 's--', color='#BF616A', label='系统动力学预测', linewidth=2, markersize=6, markerfacecolor='white', markeredgewidth=2)
        ax_top.set_title(f'{token} - 语义曲率 K(t)', fontsize=12, fontweight='bold', color='#2E3440', pad=10)
        ax_top.set_ylabel('曲率值', fontsize=10, color='#4C566A')
        ax_top.legend(fontsize=9, frameon=True, fancybox=True, shadow=True, loc='upper right')
        ax_top.grid(True, alpha=0.3, linestyle='--', color='#D8DEE9')
        ax_top.tick_params(axis='both', which='major', labelsize=9, colors='#4C566A')
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)

        # 下半部分：曲率变化率 - 使用更美观的样式
        ax_bottom = ax.inset_axes([0.08, 0.05, 0.88, 0.4])  # 调整边距
        ax_bottom.plot(layers, actual_dKdt, 'o-', color='#5E81AC', label='实际观测', linewidth=2, markersize=6, markerfacecolor='white', markeredgewidth=2)
        ax_bottom.plot(layers, pred_dKdt, 's--', color='#BF616A', label='系统动力学预测', linewidth=2, markersize=6, markerfacecolor='white', markeredgewidth=2)
        ax_bottom.set_title(f'{token} - 曲率演化速率', fontsize=12, fontweight='bold', color='#2E3440', pad=10)
        ax_bottom.set_xlabel('Transformer层', fontsize=10, color='#4C566A')
        ax_bottom.set_ylabel('变化率', fontsize=10, color='#4C566A')
        ax_bottom.legend(fontsize=9, frameon=True, fancybox=True, shadow=True, loc='upper right')
        ax_bottom.grid(True, alpha=0.3, linestyle='--', color='#D8DEE9')
        ax_bottom.tick_params(axis='both', which='major', labelsize=9, colors='#4C566A')
        ax_bottom.spines['top'].set_visible(False)
        ax_bottom.spines['right'].set_visible(False)

        # 在子图底部添加误差信息 - 美化版本
        mae_color = '#A3BE8C' if metrics['MAE'] < 0.0005 else '#EBCB8B' if metrics['MAE'] < 0.001 else '#BF616A'
        ax.text(0.5, 0.02, f'MAE: {metrics["MAE"]:.6f}', ha='center', va='bottom',
                transform=ax.transAxes, fontsize=10, fontweight='bold', color=mae_color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=mae_color, linewidth=2, alpha=0.9))

        # 隐藏主轴
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # 隐藏多余的子图
    for idx in range(n_tokens, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.2)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"🎨 现代化美化综合对比图已保存到: {save_path}")

# ====================== 主函数 ======================
def main():
    # 设置路径
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(ROOT, 'sd', 'feedback', 'layers_0_23_batch_core_variables.csv')
    output_dir = os.path.join(ROOT, 'sd', 'reinforcing_balancing_loops')

    # ==================== 参数优化 ====================
    print("="*70)
    print("参数优化阶段")
    print("="*70)

    # 使用真实数据优化参数
    try:
        optimized_params, final_mae = optimize_parameters(data_path)  # 使用所有句子

        # 更新全局参数
        global R1_GAIN, B1_SEMANTIC_GAIN, B2_DISTANCE_GAIN
        R1_GAIN = optimized_params[0]
        B1_SEMANTIC_GAIN = optimized_params[1]
        B2_DISTANCE_GAIN = optimized_params[2]

        print(f"\\n参数已更新为优化值！")
        print(f"优化后MAE: {final_mae:.6f}")

    except Exception as e:
        print(f"参数优化失败，使用默认参数: {e}")
        # 保持默认参数不变

    print("\\n" + "="*70)
    print("模型预测阶段")
    print("="*70)

    # 读取数据
    df = pd.read_csv(data_path)

    # 测试句子列表
    test_sentences = [
        '江河湖海都是水',
        "牛在田里吃草。",
        "羊喜欢在山坡上活动。",
        "鱼在水里游来游去。",
        "鸟在树枝上唱歌。",
        "马在草原上奔跑。",
        "虎在林中休息。"
    ]

    # 为每个句子进行预测
    for sentence in test_sentences:
        print(f"\n{'='*70}")
        print(f"处理句子: {sentence}")
        print(f"{'='*70}")

        df_sentence = df[df['句子'] == sentence]

        # 检查句子是否存在于数据中
        if df_sentence.empty:
            print(f"警告: 句子 '{sentence}' 在数据中不存在，跳过")
            continue

        # 获取所有token
        tokens = df_sentence['Token'].unique()

        print(f"测试句子: {sentence}")
        print(f"预测层范围: 12-16")
        print("回路参数:")
        print(f"  增强回路增益 R1 = {R1_GAIN}")
        print(f"  调节回路B1增益 (语义) = {B1_SEMANTIC_GAIN}")
        print(f"  调节回路B2增益 (距离) = {B2_DISTANCE_GAIN}")
        print(f"  反馈延迟 = {FEEDBACK_DELAY}")
        print()

        # 为每个token进行预测
        all_results = []
        for token in tokens:  # 处理所有token
            print(f"\n处理token: {token}")

            token_data = df_sentence[df_sentence['Token'] == token]
            prediction = loop_based_predict(token_data)

            if prediction is None:
                print(f"  跳过token {token} (缺少第9层数据)")
                continue

            # 分析回路动态
            analyze_loop_dynamics(token_data, prediction)

            # 评估预测质量
            metrics = evaluate_predictions(token_data, prediction)

            print(f"  预测误差: MSE={metrics['MSE']:.6f}, MAE={metrics['MAE']:.6f}, MAPE={metrics['MAPE']:.3f}")

            all_results.append({
                'token': token,
                'metrics': metrics,
                'prediction': prediction,
                'actual_data': token_data
            })

        # 生成综合图表（所有token在一张图上）
        if all_results:
            # 创建包含句子和层范围的图片文件名
            sentence_clean = sentence.replace('。', '').replace('，', '').replace(' ', '_')[:20]  # 清理句子用于文件名
            combined_plot_path = os.path.join(output_dir, f'{sentence_clean}_layers_{12}_{16}_comparison.png')
            plot_all_tokens_comparison(all_results, combined_plot_path)

    # 保存模型参数
    model_config = {
        'model_type': 'reinforcing_balancing_loops_system_dynamics',
        'parameters': {
            'R1_GAIN': R1_GAIN,
            'B1_SEMANTIC_GAIN': B1_SEMANTIC_GAIN,
            'B1_SEMANTIC_DELAY': B1_SEMANTIC_DELAY,
            'B2_DISTANCE_GAIN': B2_DISTANCE_GAIN,
            'B2_DISTANCE_THRESHOLD': B2_DISTANCE_THRESHOLD,
            'FEEDBACK_DELAY': FEEDBACK_DELAY
        },
        'description': '基于系统动力学增强回路和调节回路组合的语义演化预测模型',
        'loop_structure': {
            'R1': '增强回路：前层聚合驱动后层聚合',
            'B1': '调节回路：语义稳定性抑制过度聚合',
            'B2': '调节回路：距离变化增加聚合阻力'
        }
    }

    config_path = os.path.join(output_dir, 'loop_model_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)

    print(f"\\n回路模型配置已保存到: {config_path}")
    print("\\n系统动力学增强-调节回路模型测试完成！")

# ====================== 评估函数 ======================
def evaluate_predictions(actual_data, predicted_data):
    """
    评估预测结果的质量
    """
    layers = predicted_data['layers']
    pred_K = predicted_data['predicted_K']

    mse = 0
    mae = 0
    mape = 0
    count = 0

    for i, layer in enumerate(layers):
        actual_row = actual_data[actual_data['层'] == layer]
        if len(actual_row) > 0:
            actual_K = actual_row['曲率 K(t)'].values[0]
            pred_K_val = pred_K[i]

            error = pred_K_val - actual_K
            mse += error ** 2
            mae += abs(error)
            mape += abs(error) / abs(actual_K) if actual_K != 0 else 0
            count += 1

    if count > 0:
        mse /= count
        mae /= count
        mape /= count

    return {
        'MSE': mse,
        'MAE': mae,
        'MAPE': mape,
        'Sample_Count': count
    }

# ====================== 参数优化功能 ======================
def load_real_data(data_path):
    """
    加载真实数据用于参数优化
    """
    df = pd.read_csv(data_path)
    print(f"加载真实数据: {len(df)} 条记录，{df['句子'].nunique()} 个句子")

    # 按句子分组处理
    sentence_groups = {}
    for sentence in df['句子'].unique():
        sentence_data = df[df['句子'] == sentence]
        sentence_groups[sentence] = sentence_data

    return sentence_groups

def objective_function(params, real_data_groups, target_sentences=None):
    """
    参数优化目标函数：最小化多个句子的平均MAE
    """
    # 解包参数
    r1_gain, b1_gain, b2_gain = params

    # 更新全局参数（临时）
    global R1_GAIN, B1_SEMANTIC_GAIN, B2_DISTANCE_GAIN
    original_r1 = R1_GAIN
    original_b1 = B1_SEMANTIC_GAIN
    original_b2 = B2_DISTANCE_GAIN

    R1_GAIN = r1_gain
    B1_SEMANTIC_GAIN = b1_gain
    B2_DISTANCE_GAIN = b2_gain

    try:
        # 如果没有指定句子，使用所有句子
        if target_sentences is None:
            target_sentences = list(real_data_groups.keys())

        total_sentences_mae = 0
        valid_sentences = 0

        for sentence in target_sentences:
            if sentence not in real_data_groups:
                continue

            sentence_data = real_data_groups[sentence]

            # 为该句子的所有token进行预测并计算平均MAE
            tokens = sentence_data['Token'].unique()[:3]  # 使用前3个token
            total_sentence_mae = 0
            token_count = 0

            for token in tokens:
                token_data = sentence_data[sentence_data['Token'] == token]

                # 运行预测
                predicted_data = loop_based_predict(token_data)
                if predicted_data is None:
                    continue

                # 计算该token的MAE
                layers = predicted_data['layers']
                pred_K = predicted_data['predicted_K']

                token_mae = 0
                count = 0

                for i, layer in enumerate(layers):
                    actual_row = token_data[token_data['层'] == layer]
                    if len(actual_row) > 0:
                        actual_K = actual_row['曲率 K(t)'].values[0]
                        pred_K_val = pred_K[i]

                        # 计算MAE（平均绝对误差）
                        mae = abs(pred_K_val - actual_K)
                        token_mae += mae
                        count += 1

                if count > 0:
                    token_mae /= count
                    total_sentence_mae += token_mae
                    token_count += 1

            if token_count > 0:
                sentence_avg_mae = total_sentence_mae / token_count
                total_sentences_mae += sentence_avg_mae
                valid_sentences += 1

        if valid_sentences > 0:
            avg_mae = total_sentences_mae / valid_sentences
        else:
            avg_mae = 1.0

        # 调试输出（只在第一次调用时显示）
        if not hasattr(objective_function, 'call_count'):
            objective_function.call_count = 0
        objective_function.call_count += 1
        if objective_function.call_count % 10 == 1:  # 每10次调用显示一次
            print(f"  目标函数调用 #{objective_function.call_count}: params=[{r1_gain:.3f}, {b1_gain:.3f}, {b2_gain:.3f}], MAE={avg_mae:.6f}")

    except Exception as e:
        print(f"预测过程中出错: {e}")
        avg_mae = 1.0
    finally:
        # 恢复原始参数
        R1_GAIN = original_r1
        B1_SEMANTIC_GAIN = original_b1
        B2_DISTANCE_GAIN = original_b2

    return avg_mae

def optimize_parameters(real_data_path, target_sentences=None, max_sentences=5):
    """
    使用真实数据优化模型参数
    支持使用多个句子进行更全面的优化
    """
    print("开始参数优化...")

    # 加载数据
    real_data_groups = load_real_data(real_data_path)

    # 选择用于优化的句子
    if target_sentences is None:
        # 使用所有可用句子进行优化
        available_sentences = list(real_data_groups.keys())
        target_sentences = available_sentences  # 使用所有句子

    print(f"使用 {len(target_sentences)} 个句子进行参数优化")

    # 参数范围（扩大范围以获得更好的优化）
    bounds = [
        (0.01, 2.0),   # R1_GAIN
        (0.01, 2.0),   # B1_SEMANTIC_GAIN
        (0.01, 2.0)    # B2_DISTANCE_GAIN
    ]

    # 初始参数
    initial_params = [R1_GAIN, B1_SEMANTIC_GAIN, B2_DISTANCE_GAIN]

    print(f"初始参数: R1={initial_params[0]:.3f}, B1={initial_params[1]:.3f}, B2={initial_params[2]:.3f}")

    # 尝试多种初始点进行优化
    initial_points = [
        [0.6, 0.3, 0.2],  # 当前默认值
        [0.8, 0.4, 0.3],  # 更高的值
        [0.4, 0.2, 0.1],  # 更低的值
        [1.0, 0.5, 0.4],  # 更高值
        [0.2, 0.1, 0.05], # 更低值
    ]

    best_result = None
    best_mape = float('inf')

    for i, init_params in enumerate(initial_points):
        print(f"\\n尝试初始点 {i+1}: {init_params}")

        try:
            result = minimize(
                objective_function,
                init_params,
                args=(real_data_groups, target_sentences),
                bounds=bounds,
                method='SLSQP',  # 改用SLSQP方法
                options={'maxiter': 100, 'disp': False}
            )

            if result.success and result.fun < best_mape:
                best_result = result
                best_mape = result.fun
                print(f"  新的最佳MAPE: {best_mape:.4f}")

        except Exception as e:
            print(f"  初始点 {i+1} 优化失败: {e}")
            continue

    if best_result is None:
        print("所有优化尝试都失败了，使用默认参数")
        final_mae = objective_function(initial_params, real_data_groups, target_sentence)
        return initial_params, final_mae

    result = best_result

    # 输出详细结果
    print(f"\\n优化详细信息:")
    print(f"  迭代次数: {result.nit}")
    print(f"  函数调用次数: {result.nfev}")
    print(f"  收敛状态: {result.success}")
    print(f"  收敛消息: {result.message}")

    # 输出结果
    optimized_params = result.x
    final_mae = result.fun

    print(f"\\n优化结果:")
    print(f"  最终MAE: {final_mae:.6f}")
    print(f"  优化参数: R1={optimized_params[0]:.4f}, B1={optimized_params[1]:.4f}, B2={optimized_params[2]:.4f}")
    print(f"  优化成功: {result.success}")

    # 检查参数变化
    param_change = np.abs(np.array(optimized_params) - np.array(initial_params))
    print(f"  参数变化: R1={param_change[0]:.6f}, B1={param_change[1]:.6f}, B2={param_change[2]:.6f}")

    if np.max(param_change) < 1e-6:
        print("  注意：参数几乎没有变化，可能已达到局部最优或需要调整优化设置")

    return optimized_params, final_mae

if __name__ == "__main__":
    main()