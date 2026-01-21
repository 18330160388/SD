"""
C(t) Alpha参数消融实验

测试不同alpha值对局部语义聚类密度C(t)的影响
alpha值范围: 0.2, 0.3, 0.4, 0.5, 0.6

实验输出:
- 统计表格
- 趋势图表
- 详细分析报告
"""

import torch
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from collections import defaultdict

# 添加项目根目录到路径
# 当前文件: experiments/C/C_alpha_ablation_experiment.py
# 项目根目录: ../../../../
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)
print(f"项目根目录: {project_root}")

# 直接导入模块
sys.path.append(os.path.join(project_root, 'SD'))
from llm_hidden_extractor import extract_hidden_states
from c_t_calculator import compute_c_t, compute_c_t_batch
from m_t_calculator import ChineseMorphExtractor, compute_m_t_full

# 设置matplotlib中文字体 - 修复中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 设置seaborn风格
sns.set_style("whitegrid")
sns.set_palette("husl")

class AlphaAblationExperiment:
    """C(t) Alpha参数消融实验"""

    def __init__(self):
        self.alpha_values = [0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9]
        self.test_sentences = [
             "江河湖海都是水"
        ]
        self.results = defaultdict(dict)

    def run_experiment(self):
        """运行消融实验"""
        print("=" * 80)
        print("C(t) Alpha参数消融实验")
        print("=" * 80)
        print(f"测试Alpha值: {self.alpha_values}")
        print(f"测试句子数量: {len(self.test_sentences)}")
        print()

        # 初始化形态特征提取器
        morph_extractor = ChineseMorphExtractor()

        for sentence_idx, sentence in enumerate(self.test_sentences):
            print(f"处理句子 {sentence_idx + 1}/{len(self.test_sentences)}: {sentence}")
            print("-" * 60)

            # 提取句子特征
            try:
                hidden_states, token_num, tokenizer, inputs, attentions = extract_hidden_states(
                    text=sentence,
                    middle_layer_idx=12,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )

                # 解码tokens
                tokens = []
                for i in range(len(inputs['input_ids'][0])):
                    token_text = tokenizer.decode([inputs['input_ids'][0][i]])
                    tokens.append(token_text)

                print(f"分词结果: {tokens}")
                print(f"Token数量: {len(tokens)}")

                # 对每个alpha值进行测试
                for alpha in self.alpha_values:
                    print(f"\n测试 Alpha = {alpha}")
                    print("-" * 30)

                    c_t_values = []
                    m_t_values = []

                    # 计算每个token的C(t)和M(t)
                    for token_idx, token_text in enumerate(tokens):
                        try:
                            h_t = hidden_states[token_idx]

                            # 计算M(t)
                            m_t_value = compute_m_t_full(
                                h_t=h_t,
                                token_text=token_text,
                                tokens=tokens,
                                token_idx=token_idx,
                                hidden_states=hidden_states,
                                layer_idx=12  # 使用第12层，与LLM特征提取一致
                            )

                            # 计算C(t)
                            c_t_value = compute_c_t(
                                h_t=h_t,
                                hidden_states=hidden_states,
                                token_idx=token_idx,
                                k=3,
                                theta=0.5,
                                alpha=alpha,
                                precomputed_m_t=m_t_value
                            )

                            c_t_values.append(c_t_value)
                            m_t_values.append(m_t_value)

                            print(f"  '{token_text}': C(t)={c_t_value:.6f}, M(t)={m_t_value:.6f}")

                        except Exception as e:
                            print(f"  错误处理token '{token_text}': {e}")
                            c_t_values.append(0.0)
                            m_t_values.append(0.0)

                    # 统计结果
                    c_t_array = np.array(c_t_values)
                    m_t_array = np.array(m_t_values)

                    stats = {
                        'sentence': sentence,
                        'sentence_idx': sentence_idx,
                        'alpha': alpha,
                        'tokens': tokens,
                        'c_t_values': c_t_values,
                        'm_t_values': m_t_values,
                        'c_t_mean': float(np.mean(c_t_array)),
                        'c_t_std': float(np.std(c_t_array)),
                        'c_t_max': float(np.max(c_t_array)),
                        'c_t_min': float(np.min(c_t_array)),
                        'c_t_range': float(np.max(c_t_array) - np.min(c_t_array)),
                        'm_t_mean': float(np.mean(m_t_array)),
                        'm_t_std': float(np.std(m_t_array)),
                    }

                    self.results[alpha][sentence] = stats

                    print(f"统计结果:")
                    print(f"  C(t) - 均值: {stats['c_t_mean']:.6f}, 标准差: {stats['c_t_std']:.6f}")
                    print(f"  C(t) - 范围: [{stats['c_t_min']:.6f}, {stats['c_t_max']:.6f}]")
                    print(f"  M(t) - 均值: {stats['m_t_mean']:.6f}, 标准差: {stats['m_t_std']:.6f}")

                print()

            except Exception as e:
                print(f"处理句子失败: {e}")
                continue

        print("实验完成！")
        return self.results

    def generate_summary_table(self):
        """生成汇总表格"""
        print("\n" + "=" * 80)
        print("实验结果汇总表")
        print("=" * 80)

        # 创建汇总数据
        summary_data = []
        for alpha in self.alpha_values:
            alpha_stats = []
            for sentence in self.test_sentences:
                if sentence in self.results[alpha]:
                    stats = self.results[alpha][sentence]
                    alpha_stats.append({
                        'sentence': sentence,
                        'c_t_mean': stats['c_t_mean'],
                        'c_t_std': stats['c_t_std'],
                        'c_t_range': stats['c_t_range'],
                        'm_t_mean': stats['m_t_mean']
                    })

            if alpha_stats:
                # 计算alpha的整体统计
                c_t_means = [s['c_t_mean'] for s in alpha_stats]
                c_t_stds = [s['c_t_std'] for s in alpha_stats]
                c_t_ranges = [s['c_t_range'] for s in alpha_stats]
                m_t_means = [s['m_t_mean'] for s in alpha_stats]

                summary_data.append({
                    'Alpha': alpha,
                    'C(t)_Mean': f"{np.mean(c_t_means):.6f}",
                    'C(t)_Std': f"{np.mean(c_t_stds):.6f}",
                    'C(t)_Range_Mean': f"{np.mean(c_t_ranges):.6f}",
                    'M(t)_Mean': f"{np.mean(m_t_means):.6f}",
                    'Sentence_Count': len(alpha_stats)
                })

        # 打印表格
        if summary_data:
            df = pd.DataFrame(summary_data)
            print(df.to_string(index=False))

            # 保存到CSV
            df.to_csv('C_alpha_ablation_summary.csv', index=False)
            print(f"\n汇总表已保存到: C_alpha_ablation_summary.csv")

        return summary_data

    def plot_results(self):
        """生成可视化图表"""
        print("\n" + "=" * 80)
        print("生成可视化图表")
        print("=" * 80)

        # 1. Alpha vs C(t)均值趋势图
        plt.figure(figsize=(12, 8))

        alphas = []
        c_t_means = []
        c_t_stds = []

        for alpha in self.alpha_values:
            alpha_c_t_means = []
            for sentence in self.test_sentences:
                if sentence in self.results[alpha]:
                    alpha_c_t_means.append(self.results[alpha][sentence]['c_t_mean'])

            if alpha_c_t_means:
                alphas.append(alpha)
                c_t_means.append(np.mean(alpha_c_t_means))
                c_t_stds.append(np.std(alpha_c_t_means))

        plt.subplot(2, 2, 1)
        plt.errorbar(alphas, c_t_means, yerr=c_t_stds, fmt='o-', capsize=5, linewidth=2, markersize=8)
        plt.xlabel('Alpha 值')
        plt.ylabel('C(t) 均值')
        plt.title('Alpha vs C(t)均值趋势')
        plt.grid(True, alpha=0.3)

        # 2. Alpha vs C(t)范围趋势图
        alphas = []
        c_t_ranges = []

        for alpha in self.alpha_values:
            alpha_c_t_ranges = []
            for sentence in self.test_sentences:
                if sentence in self.results[alpha]:
                    alpha_c_t_ranges.append(self.results[alpha][sentence]['c_t_range'])

            if alpha_c_t_ranges:
                alphas.append(alpha)
                c_t_ranges.append(np.mean(alpha_c_t_ranges))

        plt.subplot(2, 2, 2)
        plt.plot(alphas, c_t_ranges, 's-', linewidth=2, markersize=8, color='orange')
        plt.xlabel('Alpha 值')
        plt.ylabel('C(t) 范围均值')
        plt.title('Alpha vs C(t)区分度')
        plt.grid(True, alpha=0.3)

        # 3. 不同句子在不同Alpha下的C(t)分布
        plt.subplot(2, 2, 3)
        sentence_names = [f"句子{i+1}" for i in range(len(self.test_sentences))]

        for i, sentence in enumerate(self.test_sentences):
            alpha_means = []
            for alpha in self.alpha_values:
                if sentence in self.results[alpha]:
                    alpha_means.append(self.results[alpha][sentence]['c_t_mean'])
                else:
                    alpha_means.append(0)

            plt.plot(self.alpha_values, alpha_means, 'o-', label=sentence_names[i],
                    linewidth=2, markersize=6)

        plt.xlabel('Alpha 值')
        plt.ylabel('C(t) 均值')
        plt.title('各句子在不同Alpha下的C(t)变化')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        # 4. Alpha参数敏感性分析
        plt.subplot(2, 2, 4)

        # 计算每个alpha的变异系数 (std/mean)
        alphas = []
        cv_values = []  # 变异系数

        for alpha in self.alpha_values:
            alpha_c_t_values = []
            for sentence in self.test_sentences:
                if sentence in self.results[alpha]:
                    alpha_c_t_values.extend(self.results[alpha][sentence]['c_t_values'])

            if alpha_c_t_values:
                alphas.append(alpha)
                mean_val = np.mean(alpha_c_t_values)
                std_val = np.std(alpha_c_t_values)
                cv = std_val / mean_val if mean_val > 0 else 0
                cv_values.append(cv)

        plt.bar(alphas, cv_values, alpha=0.7, color='green')
        plt.xlabel('Alpha 值')
        plt.ylabel('变异系数 (Std/Mean)')
        plt.title('Alpha参数敏感性分析')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('C_alpha_ablation_plots.png', dpi=300, bbox_inches='tight')
        print("图表已保存到: C_alpha_ablation_plots.png")

        # 显示图表
        plt.show()

    def analyze_optimal_alpha(self):
        """分析最优Alpha值"""
        print("\n" + "=" * 80)
        print("最优Alpha值分析")
        print("=" * 80)

        # 重新设计评分机制：考虑形态校正的合理性和平衡性
        alpha_scores = {}

        for alpha in self.alpha_values:
            scores = []

            for sentence in self.test_sentences:
                if sentence in self.results[alpha]:
                    stats = self.results[alpha][sentence]

                    # 计算基础密度（当alpha=0时的C(t)均值，作为基准）
                    base_c_t_mean = None
                    if 0.2 in self.results and sentence in self.results[0.2]:
                        # 使用最小alpha值作为基准（形态校正最小）
                        base_stats = self.results[0.2][sentence]
                        base_c_t_mean = base_stats['c_t_mean']

                    if base_c_t_mean is None:
                        # 如果没有基准，使用当前值的80%作为保守估计
                        base_c_t_mean = stats['c_t_mean'] * 0.8

                    # 1. 形态校正增益：相对于基础密度的提升（权重0.3）
                    # 不应该过度放大，但要有合理提升
                    correction_gain = (stats['c_t_mean'] - base_c_t_mean) / (base_c_t_mean + 1e-8)
                    gain_score = min(correction_gain, 0.5)  # 限制最大增益，避免过度放大

                    # 2. 数值合理性：C(t)值应该在合理范围内（权重0.3）
                    # 理想范围：[0.1, 1.0]，过大或过小都不好
                    mean_reasonable = 1.0 - abs(stats['c_t_mean'] - 0.5) / 0.5  # 0.5为中心
                    range_reasonable = min(stats['c_t_range'], 2.0) / 2.0  # 范围不应该过大
                    reasonable_score = (mean_reasonable + range_reasonable) / 2.0

                    # 3. 稳定性：标准差不应该过大（权重0.2）
                    stability_score = 1.0 / (1.0 + stats['c_t_std'])

                    # 4. Alpha合理性惩罚：过大的Alpha应该被惩罚（权重0.2）
                    # Alpha在[0.2, 0.6]范围内，0.4左右可能是最合理的
                    alpha_penalty = 1.0 - abs(alpha - 0.4) / 0.4  # 0.4为中心
                    alpha_penalty = max(alpha_penalty, 0.1)  # 最小惩罚

                    # 综合评分
                    score = (gain_score * 0.3 +
                           reasonable_score * 0.3 +
                           stability_score * 0.2 +
                           alpha_penalty * 0.2)

                    scores.append(score)

            if scores:
                alpha_scores[alpha] = np.mean(scores)

        # 排序并输出
        sorted_alphas = sorted(alpha_scores.items(), key=lambda x: x[1], reverse=True)

        print("Alpha值综合评分 (考虑平衡性和合理性):")
        print("-" * 50)
        for alpha, score in sorted_alphas:
            print(".2f")

        optimal_alpha = sorted_alphas[0][0]
        print(f"\n推荐最优Alpha值: {optimal_alpha}")
        print("选择理由:")
        print(f"- 综合评分最高 ({alpha_scores[optimal_alpha]:.4f})")
        print("- 在形态校正增益、数值合理性和稳定性间取得最佳平衡")
        print("- 避免过度放大形态校正导致的不合理聚类密度")

        return optimal_alpha

def main():
    """主函数"""
    experiment = AlphaAblationExperiment()

    # 运行实验
    results = experiment.run_experiment()

    # 生成汇总表格
    summary = experiment.generate_summary_table()

    # 生成图表
    experiment.plot_results()

    # 分析最优参数
    optimal_alpha = experiment.analyze_optimal_alpha()

    print(f"\n🎯 实验完成！最优Alpha值: {optimal_alpha}")
    print("输出文件:")
    print("- C_alpha_ablation_summary.csv (汇总表格)")
    print("- C_alpha_ablation_plots.png (可视化图表)")

if __name__ == "__main__":
    main()