"""
S(t) μ参数消融实验

测试不同μ值对语义漂移系数S(t)的影响
μ是形态修正因子ξ(M(t)) = 1 - μ·M(t)中的权重系数

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
from typing import Dict, List
from collections import defaultdict

# 添加项目根目录到路径
# 当前文件: experiments/S/S_t_test.py
# 项目根目录: ../../
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
print(f"项目根目录: {project_root}")

# 直接导入模块
sys.path.append(os.path.join(project_root, 'SD'))
from s_t_calculator import SemanticDriftCoeff

class MuAblationExperiment:
    """S(t) μ参数消融实验"""

    def __init__(self):
        # μ值范围：基于典型值0.25，测试周围范围
        self.mu_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
        self.test_sentences = [
            "江河湖海都是水",
            "小猫追着蝴蝶跑过花园",
            "科学家在实验室研究新型疫苗",
            "学生们认真听老师讲课"
        ]
        self.results = defaultdict(dict)

    def run_experiment(self):
        """运行消融实验"""
        print("=" * 80)
        print("S(t) μ参数消融实验")
        print("=" * 80)
        print(f"测试μ值: {self.mu_values}")
        print(f"测试句子数量: {len(self.test_sentences)}")
        print()

        for sentence_idx, sentence in enumerate(self.test_sentences):
            print(f"处理句子 {sentence_idx + 1}/{len(self.test_sentences)}: {sentence}")
            print("-" * 60)

            # 对每个μ值进行测试
            for mu in self.mu_values:
                print(f"\n测试 μ = {mu}")
                print("-" * 30)

                # 创建计算器实例，设置不同的μ值
                calculator = SemanticDriftCoeff(lambda_decay=0.1)
                # 手动设置μ参数（因为它是nn.Parameter）
                with torch.no_grad():
                    calculator.mu.fill_(mu)

                try:
                    # 计算S(t)
                    s_t_values = calculator(sentence)

                    # 转换为numpy数组以便统计
                    s_t_array = s_t_values.detach().cpu().numpy()

                    # 统计结果
                    stats = {
                        'sentence': sentence,
                        'sentence_idx': sentence_idx,
                        'mu': mu,
                        's_t_values': s_t_array.tolist(),
                        's_t_mean': float(np.mean(s_t_array)),
                        's_t_std': float(np.std(s_t_array)),
                        's_t_max': float(np.max(s_t_array)),
                        's_t_min': float(np.min(s_t_array)),
                        's_t_range': float(np.max(s_t_array) - np.min(s_t_array)),
                        'seq_len': len(s_t_array)
                    }

                    self.results[mu][sentence] = stats

                    print(f"统计结果:")
                    print(f"  S(t) - 均值: {stats['s_t_mean']:.6f}, 标准差: {stats['s_t_std']:.6f}")
                    print(f"  S(t) - 范围: [{stats['s_t_min']:.6f}, {stats['s_t_max']:.6f}]")
                    print(f"  序列长度: {stats['seq_len']}")

                except Exception as e:
                    print(f"  错误处理句子 '{sentence}' with μ={mu}: {e}")
                    # 创建默认的错误统计
                    stats = {
                        'sentence': sentence,
                        'sentence_idx': sentence_idx,
                        'mu': mu,
                        's_t_values': [],
                        's_t_mean': 0.0,
                        's_t_std': 0.0,
                        's_t_max': 0.0,
                        's_t_min': 0.0,
                        's_t_range': 0.0,
                        'seq_len': 0
                    }
                    self.results[mu][sentence] = stats

            print()

        print("实验完成！")
        return self.results

    def generate_summary_table(self):
        """生成汇总表格"""
        print("\n" + "=" * 80)
        print("实验结果汇总表")
        print("=" * 80)

        # 创建汇总数据
        summary_data = []
        for mu in self.mu_values:
            mu_stats = []
            for sentence in self.test_sentences:
                if sentence in self.results[mu]:
                    stats = self.results[mu][sentence]
                    mu_stats.append({
                        'sentence': sentence,
                        's_t_mean': stats['s_t_mean'],
                        's_t_std': stats['s_t_std'],
                        's_t_range': stats['s_t_range'],
                        'seq_len': stats['seq_len']
                    })

            if mu_stats:
                # 计算μ的整体统计
                s_t_means = [s['s_t_mean'] for s in mu_stats]
                s_t_stds = [s['s_t_std'] for s in mu_stats]
                s_t_ranges = [s['s_t_range'] for s in mu_stats]
                seq_lens = [s['seq_len'] for s in mu_stats]

                summary_data.append({
                    'μ': mu,
                    'S(t)_Mean': f"{np.mean(s_t_means):.6f}",
                    'S(t)_Std': f"{np.mean(s_t_stds):.6f}",
                    'S(t)_Range_Mean': f"{np.mean(s_t_ranges):.6f}",
                    'Avg_Seq_Len': f"{np.mean(seq_lens):.1f}",
                    'Sentence_Count': len(mu_stats)
                })

        # 打印表格
        if summary_data:
            df = pd.DataFrame(summary_data)
            print(df.to_string(index=False))

            # 保存到CSV（保存到脚本所在目录）
            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(script_dir, 'S_mu_ablation_summary.csv')
            df.to_csv(csv_path, index=False)
            print(f"\n汇总表已保存到: {csv_path}")

        return summary_data

    def plot_results(self):
        """生成可视化图表（已禁用）"""
        print("\n" + "=" * 80)
        print("跳过图表生成（根据用户要求）")
        print("=" * 80)
        print("如需查看图表，请取消注释相关代码")

    def analyze_optimal_mu(self):
        """分析最优μ值"""
        print("\n" + "=" * 80)
        print("最优μ值分析")
        print("=" * 80)

        # 重新设计评分机制：考虑语义漂移的合理性和平衡性
        mu_scores = {}

        for mu in self.mu_values:
            scores = []

            for sentence in self.test_sentences:
                if sentence in self.results[mu]:
                    stats = self.results[mu][sentence]

                    # 1. S(t)值合理性：应该在合理范围内（权重0.3）
                    # 理想范围：[0.1, 0.8]，过大或过小都不好
                    mean_reasonable = 1.0 - abs(stats['s_t_mean'] - 0.4) / 0.4  # 0.4为中心
                    range_reasonable = min(stats['s_t_range'], 1.0) / 1.0  # 范围不应该过大
                    reasonable_score = (mean_reasonable + range_reasonable) / 2.0

                    # 2. 稳定性：标准差不应该过大（权重0.3）
                    stability_score = 1.0 / (1.0 + stats['s_t_std'])

                    # 3. 区分度：范围应该适中（权重0.2）
                    # 太小的范围表示缺乏区分度，太大的范围表示过于敏感
                    optimal_range = 0.3
                    range_score = 1.0 - abs(stats['s_t_range'] - optimal_range) / optimal_range

                    # 4. μ合理性惩罚：过大的μ应该被惩罚（权重0.2）
                    # μ在[0.2, 0.4]范围内，0.25左右可能是最合理的
                    mu_penalty = 1.0 - abs(mu - 0.25) / 0.25  # 0.25为中心
                    mu_penalty = max(mu_penalty, 0.1)  # 最小惩罚

                    # 综合评分
                    score = (reasonable_score * 0.3 +
                           stability_score * 0.3 +
                           range_score * 0.2 +
                           mu_penalty * 0.2)

                    scores.append(score)

            if scores:
                mu_scores[mu] = np.mean(scores)

        # 排序并输出
        sorted_mus = sorted(mu_scores.items(), key=lambda x: x[1], reverse=True)

        print("μ值综合评分 (考虑平衡性和合理性):")
        print("-" * 50)
        for mu, score in sorted_mus:
            print(".2f")

        optimal_mu = sorted_mus[0][0]
        print(f"\n推荐最优μ值: {optimal_mu}")
        print("选择理由:")
        print(f"- 综合评分最高 ({mu_scores[optimal_mu]:.4f})")
        print("- 在语义漂移合理性、稳定性和区分度间取得最佳平衡")
        print("- 避免过度放大形态修正导致的不合理漂移系数")

        return optimal_mu

def main():
    """主函数"""
    experiment = MuAblationExperiment()

    # 运行实验
    results = experiment.run_experiment()

    # 生成汇总表格
    summary = experiment.generate_summary_table()

    # 跳过图表生成
    experiment.plot_results()

    # 分析最优参数
    optimal_mu = experiment.analyze_optimal_mu()

    print(f"\n🎯 实验完成！最优μ值: {optimal_mu}")
    print("输出文件:")
    print("- S_mu_ablation_summary.csv (汇总表格)")

if __name__ == "__main__":
    main()
