import numpy as np
import torch
import pandas as pd
from typing import List, Dict, Tuple
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from llm_hidden_extractor import extract_hidden_states
from d_t_calculator import compute_d_t_batch
from m_t_calculator import compute_m_t_full

def load_test_sentences() -> List[str]:
    """
    加载测试句子，包含形态相关的中文词对
    """
    sentences = [
        "小猫追着蝴蝶跑过花园",  # 动物+动作+动物+动作+地点
        "阳光洒在金色的沙滩上",  # 自然现象+介词+形容词+名词+介词
        "学生认真阅读一本好书",  # 人称+副词+动作+数量词+形容词+名词
        "医生仔细检查病人病情",  # 职业+副词+动作+名词+名词
        "老师耐心讲解数学题",   # 职业+形容词+动作+名词+名词
        "工人努力建造新房子",   # 职业+副词+动作+形容词+名词
        "农民辛苦种植水稻田",   # 职业+形容词+动作+名词+名词
        "司机安全驾驶汽车跑",   # 职业+形容词+动作+名词+动作
        "厨师精心烹饪美味菜",   # 职业+副词+动作+形容词+名词
        "画家细致绘制风景画"    # 职业+形容词+动作+名词+名词
    ]
    return sentences

def calculate_distance_discrimination(D_t_values: np.ndarray, tokens: List[str]) -> float:
    """
    计算距离区分度：形态相关词对的D(t)差异平均值

    形态相关词对定义：
    - 同义词或语义相关词：如"认真"+"仔细"，"耐心"+"细致"等
    - 职业+动作相关：如"医生"+"检查"，"老师"+"讲解"等
    """
    # 定义形态相关词对（基于句子内容）
    related_pairs = [
        # 副词相关
        ("认真", "仔细"), ("耐心", "细致"), ("努力", "辛苦"), ("安全", "精心"),
        # 职业+动作相关
        ("医生", "检查"), ("老师", "讲解"), ("工人", "建造"), ("农民", "种植"),
        ("司机", "驾驶"), ("厨师", "烹饪"), ("画家", "绘制"),
        # 形容词相关
        ("好", "美味"), ("新", "水稻"), ("金色", "沙滩")
    ]

    total_diff = 0.0
    pair_count = 0

    for token1, token2 in related_pairs:
        if token1 in tokens and token2 in tokens:
            idx1 = tokens.index(token1)
            idx2 = tokens.index(token2)
            diff = abs(D_t_values[idx1] - D_t_values[idx2])
            total_diff += diff
            pair_count += 1

    if pair_count == 0:
        return 0.0

    return total_diff / pair_count

def run_alpha_ablation_experiment(alpha_values: List[float] = [0.0, 0.2, 0.3, 0.4, 0.5]) -> pd.DataFrame:
    """
    运行α消融实验

    Args:
        alpha_values: 测试的α值列表

    Returns:
        results_df: 实验结果DataFrame
    """
    sentences = load_test_sentences()
    results = []

    print("开始α消融实验...")
    print(f"测试句子数量: {len(sentences)}")
    print(f"测试α值: {alpha_values}")
    print("=" * 80)

    for sentence in sentences:
        print(f"\n处理句子: {sentence}")

        try:
            # 提取隐藏状态
            h_t, token_num, tokenizer, inputs, _ = extract_hidden_states(sentence, middle_layer_idx=12)
            tokens = []
            for token_id in inputs['input_ids'][0]:
                token_text = tokenizer.decode([token_id])
                if token_text not in ['<|endoftext|>', '<|im_start|>', '<|im_end|>']:
                    tokens.append(token_text)

            if len(tokens) != token_num:
                print(f"  跳过: token数量不匹配")
                continue

            # 计算M(t) - 形态匹配度
            m_t_list = []
            for i in range(token_num):
                try:
                    m_t = compute_m_t_full(h_t[i], sentence, tokens, i, h_t, layer_idx=12)
                    m_t_list.append(m_t)
                except:
                    m_t_list.append(0.5)  # 默认值
            m_t_array = np.array(m_t_list)

            # 对每个α值计算D(t)和距离区分度
            for alpha in alpha_values:
                D_t_values = compute_d_t_batch(
                    hidden_states=h_t,
                    window_size=3,
                    sim_threshold=0.5,
                    epsilon=1e-6,
                    precomputed_m_t_list=m_t_array,
                    alpha=alpha,
                    return_diagnostics=False
                )

                discrimination = calculate_distance_discrimination(D_t_values, tokens)

                results.append({
                    'sentence': sentence,
                    'alpha': alpha,
                    'tokens': tokens,
                    'D_t_values': D_t_values.tolist(),
                    'avg_D_t': np.mean(D_t_values),
                    'std_D_t': np.std(D_t_values),
                    'distance_discrimination': discrimination
                })

                print(f"  α={alpha:.1f}: 平均D(t)={np.mean(D_t_values):.4f}, 区分度={discrimination:.4f}")

        except Exception as e:
            print(f"  错误: {e}")
            continue

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    # 计算总体统计
    summary = []
    for alpha in alpha_values:
        alpha_data = results_df[results_df['alpha'] == alpha]
        if len(alpha_data) > 0:
            avg_discrimination = alpha_data['distance_discrimination'].mean()
            summary.append({
                'alpha': alpha,
                'avg_distance_discrimination': avg_discrimination,
                'sentence_count': len(alpha_data)
            })

    summary_df = pd.DataFrame(summary)

    # 计算相对下降（相对于α=0）
    if 0.0 in summary_df['alpha'].values:
        baseline = summary_df[summary_df['alpha'] == 0.0]['avg_distance_discrimination'].iloc[0]
        summary_df['relative_drop'] = (baseline - summary_df['avg_distance_discrimination']) / baseline * 100

    print("\n" + "=" * 80)
    print("实验总结:")
    print(summary_df.to_string(index=False, float_format='%.4f'))

    return results_df, summary_df

def save_results(results_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str = "experiments/D"):
    """
    保存实验结果
    """
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, "alpha_ablation_results.csv")
    summary_file = os.path.join(output_dir, "alpha_ablation_summary.csv")

    results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')

    print(f"\n结果已保存到:")
    print(f"  {results_file}")
    print(f"  {summary_file}")

if __name__ == "__main__":
    # 运行实验
    alpha_values = [0.0, 0.2, 0.3, 0.4, 0.5]  # 包括无修正的情况
    results_df, summary_df = run_alpha_ablation_experiment(alpha_values)

    # 保存结果
    save_results(results_df, summary_df)

    print("\n实验完成！")
    print("关键发现:")
    print("- α=0: 无形态修正的基准")
    print("- α>0: 应用形态修正，预期降低形态相关词的距离区分度")
    print("- 相对下降≥10%: 验证形态修正的有效性")