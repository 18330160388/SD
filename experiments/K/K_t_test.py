"""
中文LLM语义层局部截面曲率 K(t) 科学验证实验

实验目标：
1. 实验1：语义收敛vs发散验证
   - 无歧义句（如"苹果是水果"）的平均Kt > 0（正曲率，语义收敛）
   - 多义句（如"他很行"）的平均Kt < 0（负曲率，语义发散）
   - 差异 ≥ 0.3（符合语义逻辑）

2. 实验2：形态强化效应验证
   - 中文形态相关句（如"松柏杨柳都是树木"）的平均Kt显著高于形态无关句（如"苹果、电脑、书籍"）
   - 差异 ≥ 0.2（体现形态对弯曲的强化）

基于k_t_calculator.py的完整实现
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Tuple, Dict
import pandas as pd
from pathlib import Path

# Ensure project root on sys.path for local imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 导入项目内工具
from k_t_calculator import compute_curvature_batch, compute_local_sectional_curvature, compute_curvature_batch_with_scaled
from llm_hidden_extractor import extract_hidden_states
from m_t_calculator import compute_m_t_full

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建输出目录
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "experiments" / "K_t_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_llm_states(text: str, layer_idx: int = 12) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """提取LLM隐藏状态的统一接口"""
    h_t, token_num, tokenizer, inputs, attn_weights = extract_hidden_states(
        text=text,
        middle_layer_idx=layer_idx
    )

    input_ids = inputs['input_ids'].squeeze(0)
    tokens = [tokenizer.decode([int(token_id)]) for token_id in input_ids]

    return h_t, tokens, attn_weights


# ==================== 实验1：语义收敛vs发散验证 ====================
def experiment_1_semantic_convergence():
    """实验1：语义收敛vs发散验证

    验证目标：
    - 无歧义句的平均Kt > 0（正曲率，语义收敛）
    - 多义句的平均Kt < 0（负曲率，语义发散）
    - 差异 ≥ 0.3
    """
    print("\n" + "="*80)
    print("实验1：语义收敛 vs 发散验证".center(80))
    print("="*80)

    # 测试样本 - 优化选择更典型的例子
    test_cases = [
        # 无歧义句（语义收敛，正曲率）
        {"text": "江河湖海都是水", "label": "无歧义"},

        # 多义句（语义发散，负曲率）
        {"text": "他很行", "label": "多义"}
    ]

    results = []

    for case in test_cases:
        text = case["text"]
        label = case["label"]

        print(f"\n处理句子：{text} [{label}]")

        # 提取LLM状态
        h_t, tokens, attn_weights = extract_llm_states(text)

        # 计算M(t)值（用于K(t)计算）
        m_t_values = []
        for t, token in enumerate(tokens):
            # 计算M(t) - 使用简化的计算（不显示详细输出）
            import io
            import contextlib
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                M_t = compute_m_t_full(
                    h_t=h_t[t],
                    token_text=token,
                    tokens=tokens,
                    token_idx=t,
                    hidden_states=h_t,
                    layer_idx=12
                )
            m_t_values.append(M_t)

        # 计算K(t)（同时返回按D缩放的K）
        k_t_values, k_t_values_scaled = compute_curvature_batch_with_scaled(
            hidden_states=h_t,
            sentence_length=len(tokens),
            window_size=3,
            precomputed_m_t_list=m_t_values,
            unit_normalize=True,
            scale_by_D=True
        )

        # 计算句子平均值（原始与缩放）
        avg_k_t = np.mean(k_t_values)
        avg_k_t_scaled = np.mean(k_t_values_scaled)

        results.append({
            "句子": text,
            "类型": label,
            "平均K(t)": avg_k_t,
            "平均K_scaled": avg_k_t_scaled,
            "Token数量": len(tokens)
        })

        print(f"  - 平均K_scaled: {avg_k_t_scaled:.4f} (基于{len(tokens)}个token)")

    # 转换为DataFrame
    df = pd.DataFrame(results)

    if len(df) == 0:
        print("\n! 错误：未能提取到任何有效数据")
        return None, {"error": "No valid data"}

    # 统计分析
    # 使用缩放后的K值进行验证判断（K_scaled）
    unambiguous_data = df[df["类型"] == "无歧义"]["平均K_scaled"]
    ambiguous_data = df[df["类型"] == "多义"]["平均K_scaled"]

    if len(unambiguous_data) == 0 or len(ambiguous_data) == 0:
        print("\n! 错误：无歧义或多义样本数据不足")
        return None, {"error": "Insufficient data"}

    unambiguous_mean = unambiguous_data.mean()
    ambiguous_mean = ambiguous_data.mean()
    difference = unambiguous_mean - ambiguous_mean  # 正值表示无歧义>多义

    print("\n" + "-"*80)
    print("【统计结果】")
    print(f"  无歧义句平均 K_scaled: {unambiguous_mean:.4f}")
    print(f"  多义句平均 K_scaled:   {ambiguous_mean:.4f}")
    print(f"  差异 (无歧义-多义) [scaled]: {difference:.4f}")

    # 验证条件（基于缩放值）
    pass_cond = (unambiguous_mean > 0) and (ambiguous_mean < 0) and (difference >= 0.3)
    pass_text = "[PASS]" if pass_cond else "[FAIL]"
    print(f"  验证结果:          {pass_text}")
    print(f"  条件检查:")
    print(f"    无歧义 > 0:       {'✓' if unambiguous_mean > 0 else '✗'} ({unambiguous_mean:.4f})")
    print(f"    多义 < 0:         {'✓' if ambiguous_mean < 0 else '✗'} ({ambiguous_mean:.4f})")
    print(f"    差异 ≥ 0.3:       {'✓' if difference >= 0.3 else '✗'} ({difference:.4f})")
    print("-"*80)

    # 保存数据
    df.to_csv(OUTPUT_DIR / "exp1_convergence_data.csv", index=False, encoding="utf-8-sig")

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1：条形图
    categories = ["无歧义句", "多义句"]
    means = [unambiguous_mean, ambiguous_mean]
    colors = ['#2ecc71', '#e74c3c']

    bars = ax1.bar(categories, means, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_ylabel('局部截面曲率 K_scaled', fontsize=12)
    ax1.set_title('实验1：语义收敛 vs 发散', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10, fontweight='bold')

    # 子图2：箱线图
    unambiguous_vals = df[df["类型"] == "无歧义"]["平均K(t)"].values
    ambiguous_vals = df[df["类型"] == "多义"]["平均K(t)"].values

    bp = ax2.boxplot([unambiguous_vals, ambiguous_vals],
                      labels=categories,
                      patch_artist=True,
                      notch=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_ylabel('局部截面曲率 K_scaled', fontsize=12)
    ax2.set_title('实验1：K(t) 分布对比', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp1_convergence_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[+] 实验1完成，结果已保存至: {OUTPUT_DIR}")

    return df, {
        "unambiguous_mean_scaled": unambiguous_mean,
        "ambiguous_mean_scaled": ambiguous_mean,
        "difference_scaled": difference
    }


# ==================== 实验2：形态强化效应验证 ====================
def experiment_2_morphological_enhancement():
    """实验2：形态强化效应验证

    验证目标：
    - 中文形态相关句的平均Kt显著高于形态无关句
    - 差异 ≥ 0.2
    """
    print("\n" + "="*80)
    print("实验2：形态强化效应验证".center(80))
    print("="*80)

    # 测试样本
    test_cases = [
        # 形态相关句（形态相似的词语组合）
        {"text": "松柏杨柳都是树木", "label": "形态相关", "description": "树木名称形态相似"},
        {"text": "桃李杏梅开满园", "label": "形态相关", "description": "水果名称形态相似"},
        {"text": "江河湖海都是水域", "label": "形态相关", "description": "水域名称形态相似"},
        {"text": "刀剑枪戟是兵器", "label": "形态相关", "description": "兵器名称形态相似"},
        {"text": "青黄赤白黑五色", "label": "形态相关", "description": "颜色名称形态相似"},
        {"text": "春夏秋冬四季更替", "label": "形态相关", "description": "季节名称形态相似"},
        {"text": "金银铜铁是金属", "label": "形态相关", "description": "金属名称形态相似"},
        {"text": "猪牛羊马是家畜", "label": "形态相关", "description": "动物名称形态相似"},

        # 形态无关句（语义相关但形态不同）
        {"text": "苹果、电脑、书籍", "label": "形态无关", "description": "物品名称形态各异"},
        {"text": "医生、老师、司机", "label": "形态无关", "description": "职业名称形态各异"},
        {"text": "吃饭、睡觉、工作", "label": "形态无关", "description": "动作名称形态各异"},
        {"text": "红色、快速、美丽", "label": "形态无关", "description": "形容词形态各异"},
        {"text": "北京、上海、广州", "label": "形态无关", "description": "地名形态各异"},
        {"text": "语文、数学、英语", "label": "形态无关", "description": "学科名称形态各异"},
        {"text": "手机、电视、电脑", "label": "形态无关", "description": "电器名称形态各异"},
        {"text": "跑步、游泳、骑车", "label": "形态无关", "description": "运动名称形态各异"},
    ]

    results = []

    for case in test_cases:
        text = case["text"]
        label = case["label"]
        description = case["description"]

        print(f"\n处理句子：{text}")
        print(f"  类型：{label} ({description})")

        # 提取LLM状态
        h_t, tokens, attn_weights = extract_llm_states(text)

        # 计算M(t)值
        m_t_values = []
        for t, token in enumerate(tokens):
            import io
            import contextlib
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                M_t = compute_m_t_full(
                    h_t=h_t[t],
                    token_text=token,
                    tokens=tokens,
                    token_idx=t,
                    hidden_states=h_t,
                    layer_idx=12
                )
            m_t_values.append(M_t)

        # 计算K(t)
        k_t_values, k_t_values_scaled = compute_curvature_batch_with_scaled(
            hidden_states=h_t,
            sentence_length=len(tokens),
            window_size=3,
            sim_threshold=0.5,
            precomputed_m_t_list=m_t_values,
            unit_normalize=True,
            scale_by_D=True
        )

        # 计算句子平均K(t)（原始与缩放）
        avg_k_t = np.mean(k_t_values)
        avg_k_t_scaled = np.mean(k_t_values_scaled)
        results.append({
            "句子": text,
            "类型": label,
            "描述": description,
            "平均K(t)": avg_k_t,
            "平均K_scaled": avg_k_t_scaled,
            "Token数量": len(tokens)
        })

        print(f"  - 平均K(t): {avg_k_t:.4f} (基于{len(tokens)}个token)")

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 统计分析（基于缩放后的K）
    morph_related_data = df[df["类型"] == "形态相关"]["平均K_scaled"]
    morph_unrelated_data = df[df["类型"] == "形态无关"]["平均K_scaled"]

    morph_related_mean = morph_related_data.mean()
    morph_unrelated_mean = morph_unrelated_data.mean()
    difference = morph_related_mean - morph_unrelated_mean  # 正值表示形态相关>无关

    print("\n" + "-"*80)
    print("【统计结果】")
    print(f"  形态相关句平均 K_scaled: {morph_related_mean:.4f}")
    print(f"  形态无关句平均 K_scaled: {morph_unrelated_mean:.4f}")
    print(f"  差异 (相关-无关) [scaled]:     {difference:.4f}")

    # 验证条件（基于缩放值）
    pass_cond = difference >= 0.2
    pass_text = "[PASS]" if pass_cond else "[FAIL]"
    print(f"  验证结果:              {pass_text}")
    print(f"  条件检查: 差异 ≥ 0.2:  {'✓' if difference >= 0.2 else '✗'} ({difference:.4f})")
    print("-"*80)

    # 保存数据
    df.to_csv(OUTPUT_DIR / "exp2_morphological_data.csv", index=False, encoding="utf-8-sig")

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1：条形图
    categories = ["形态相关", "形态无关"]
    means = [morph_related_mean, morph_unrelated_mean]
    colors = ['#3498db', '#e67e22']

    bars = ax1.bar(categories, means, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('平均局部截面曲率 K_scaled', fontsize=12)
    ax1.set_title('实验2：形态强化效应', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 子图2：箱线图
    morph_related_vals = df[df["类型"] == "形态相关"]["平均K(t)"].values
    morph_unrelated_vals = df[df["类型"] == "形态无关"]["平均K(t)"].values

    bp = ax2.boxplot([morph_related_vals, morph_unrelated_vals],
                      labels=categories,
                      patch_artist=True,
                      notch=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel('平均局部截面曲率 K_scaled', fontsize=12)
    ax2.set_title('实验2：K(t) 分布对比', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp2_morphological_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[+] 实验2完成，结果已保存至: {OUTPUT_DIR}")

    return df, {
        "morph_related_mean_scaled": morph_related_mean,
        "morph_unrelated_mean_scaled": morph_unrelated_mean,
        "difference_scaled": difference
    }


# ==================== 主实验运行器 ====================
def run_all_experiments():
    """运行所有K(t)验证实验"""
    print("\n" + "="*80)
    print("中文LLM语义层局部截面曲率 K(t) 科学验证实验".center(80))
    print("="*80)
    print(f"\n输出目录: {OUTPUT_DIR}")
    print("\n包含2个实验：")
    print("  1. 语义收敛 vs 发散验证")
    print("  2. 形态强化效应验证")

    results_summary = {}

    # 实验1
    try:
        _, stats1 = experiment_1_semantic_convergence()
        results_summary["实验1"] = stats1
    except Exception as e:
        print(f"\n[X] 实验1失败: {e}")
        results_summary["实验1"] = {"error": str(e)}

    # 实验2
    try:
        _, stats2 = experiment_2_morphological_enhancement()
        results_summary["实验2"] = stats2
    except Exception as e:
        print(f"\n[X] 实验2失败: {e}")
        results_summary["实验2"] = {"error": str(e)}

    # 生成总结报告
    generate_summary_report(results_summary)

    print("\n" + "="*80)
    print("所有实验完成！".center(80))
    print(f"结果已保存至: {OUTPUT_DIR}".center(80))
    print("="*80)


def generate_summary_report(results_summary: Dict):
    """生成实验总结报告"""
    print("\n" + "="*80)
    print("K(t) 验证实验总结报告".center(80))
    print("="*80)

    summary_data = []

    # 实验1（使用缩放后的K评判）
    if "error" not in results_summary.get("实验1", {}):
        exp1 = results_summary["实验1"]
        unamb_mean = exp1.get('unambiguous_mean_scaled', exp1.get('unambiguous_mean', 0))
        amb_mean = exp1.get('ambiguous_mean_scaled', exp1.get('ambiguous_mean', 0))
        diff = exp1.get('difference_scaled', exp1.get('difference', 0))
        pass_cond1 = (unamb_mean > 0 and amb_mean < 0 and diff >= 0.3)
        summary_data.append({
            "实验编号": "实验1",
            "实验名称": "语义收敛vs发散",
            "验证指标": "无歧义>0，多义<0，差异≥0.3 (基于K_scaled)",
            "实际值": f"无歧义={unamb_mean:.3f}, 多义={amb_mean:.3f}, 差异={diff:.3f}",
            "验证结果": "[PASS]" if pass_cond1 else "[FAIL]"
        })

    # 实验2（使用缩放后的K评判）
    if "error" not in results_summary.get("实验2", {}):
        exp2 = results_summary["实验2"]
        diff2 = exp2.get('difference_scaled', exp2.get('difference', 0))
        pass_cond2 = diff2 >= 0.2
        summary_data.append({
            "实验编号": "实验2",
            "实验名称": "形态强化效应",
            "验证指标": "形态相关-无关差异≥0.2 (基于K_scaled)",
            "实际值": f"差异={diff2:.3f}",
            "验证结果": "[PASS]" if pass_cond2 else "[FAIL]"
        })

    # 保存总结表格
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(OUTPUT_DIR / "summary_report.csv", index=False, encoding="utf-8-sig")

    # 打印总结表格
    print("\n")
    print(df_summary.to_string(index=False))
    print("\n")

    # 生成总结可视化
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#2ecc71' if '[PASS]' in row["验证结果"] else '#e74c3c'
              for _, row in df_summary.iterrows()]

    y_pos = np.arange(len(df_summary))
    ax.barh(y_pos, [1]*len(df_summary), color=colors, alpha=0.7, edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['实验编号']}\n{row['实验名称']}" for _, row in df_summary.iterrows()])
    ax.set_xlim(0, 1)
    ax.set_xlabel('验证状态', fontsize=12)
    ax.set_title('K(t) 验证实验总结', fontsize=16, fontweight='bold')
    ax.set_xticks([])

    # 添加结果标签
    for i, (_, row) in enumerate(df_summary.iterrows()):
        ax.text(0.5, i, f"{row['验证结果']}\n{row['实际值'][:20]}...",
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "summary_report_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[+] 总结报告已保存至: {OUTPUT_DIR / 'summary_report.csv'}")


if __name__ == "__main__":
    run_all_experiments()