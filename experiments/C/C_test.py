"""
C(t) 局部语义聚类密度计算验证程序

使用真实句子和LLM提取的特征，验证c_t_calculator.py中的C(t)计算的正确性
"""

import torch
import sys
import os
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from llm_hidden_extractor import extract_hidden_states
from c_t_calculator import compute_c_t, compute_c_t_batch
from m_t_calculator import ChineseMorphExtractor, compute_m_t  # 修正导入

def test_c_t_calculation():
    """测试C(t)局部语义聚类密度计算"""

    print("=" * 60)
    print("C(t) 局部语义聚类密度计算验证")
    print("=" * 60)

    # 1. 准备测试句子
    test_sentences = [
        "他打篮球打得很好",  # "打"有多义
        "他打了小明一下",   # "打"有不同义项
        "这个计划行不通",   # "行"有多义
    ]

    # 选择第一个句子进行详细测试
    test_text = test_sentences[0]
    print(f"测试句子: {test_text}")
    print()

    # 2. 使用真实LLM提取特征
    print("正在提取LLM特征...")
    try:
        hidden_states, token_num, tokenizer, inputs, attentions = extract_hidden_states(
            text=test_text,
            middle_layer_idx=12,  # 使用第12层
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"✓ 成功提取特征")
        print(f"  - Token数量: {token_num}")
        print(f"  - Hidden维度: {hidden_states.shape[1]}")
    except Exception as e:
        print(f"✗ 特征提取失败: {e}")
        return

    # 3. 解码tokens并修复中文显示
    raw_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # 使用tokenizer的decode方法获取可读文本
    decoded_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    print(f"原始句子: {test_text}")
    print(f"解码文本: {decoded_text}")

    # 显示token信息
    tokens = []
    for i, token_id in enumerate(inputs['input_ids'][0]):
        token_text = tokenizer.decode([token_id])
        tokens.append(token_text)

    print(f"分词结果: {tokens}")
    print()

    # 4. 初始化M(t)计算器
    print("初始化M(t)计算器...")
    morph_extractor = ChineseMorphExtractor()

    # 5. 计算每个token的C(t)
    print("计算C(t)局部语义聚类密度...")
    print("说明：")
    print("- C(t) = [N_eff(t) / (V_local(t)+eps)] * γ(M(t))")
    print("- N_eff(t): 语义相似度 >= θ 的有效向量数")
    print("- V_local(t): 局部子空间体积（协方差椭圆体）")
    print("- γ(M(t)): 形态修正因子")
    print("-" * 40)

    results = []
    for i, token_text in enumerate(tokens):
        try:
            # 获取当前token的h(t)
            h_t = hidden_states[i]

            # 计算M(t)值
            m_t_value = compute_m_t(
                h_t=h_t,
                token_text=token_text
            )

            # 计算C(t)
            c_t_value = compute_c_t(
                h_t=h_t,
                hidden_states=hidden_states,
                token_idx=i,
                k=3,  # 上下文窗口大小
                theta=0.5,  # 语义相似度阈值
                alpha=0.4,  # 形态修正因子权重
                precomputed_m_t=m_t_value.item() if torch.is_tensor(m_t_value) else m_t_value
            )

            print(f"Token '{token_text}' (位置{i}):")
            print(f"  C(t) = {c_t_value:.6f}")
            print(f"  M(t) = {m_t_value.item() if torch.is_tensor(m_t_value) else m_t_value:.6f}")
            print()

            results.append({
                'token': token_text,
                'position': i,
                'c_t': c_t_value,
                'm_t': m_t_value.item() if torch.is_tensor(m_t_value) else m_t_value
            })

        except Exception as e:
            print(f"✗ 计算token '{token_text}'失败: {e}")
            continue

    # 6. 批量计算验证
    print("批量计算验证:")
    print("-" * 40)

    # 准备M(t)数组
    m_t_values = []
    for i, token_text in enumerate(tokens):
        try:
            m_t_val = compute_m_t(
                h_t=hidden_states[i],
                token_text=token_text
            )
            m_t_values.append(m_t_val)
        except:
            m_t_values.append(0.5)  # 默认值

    m_t_array = np.array(m_t_values)

    # 批量计算C(t)
    c_t_batch = compute_c_t_batch(
        hidden_states=hidden_states,
        k=3,
        theta=0.5,
        alpha=0.4,
        precomputed_m_t_list=m_t_array
    )

    print(f"批量计算结果: {c_t_batch}")

    # 验证单次和批量计算的一致性
    single_results = np.array([r['c_t'] for r in results])
    consistency = np.allclose(single_results, c_t_batch[:len(single_results)], rtol=1e-6)
    print(f"✓ 单次vs批量计算一致性: {consistency}")

    # 7. 验证结果合理性
    print()
    print("验证结果合理性:")
    print("-" * 40)

    c_t_values = [r['c_t'] for r in results]
    print(f"✓ C(t)值范围: [{min(c_t_values):.6f}, {max(c_t_values):.6f}]")
    print(f"✓ 所有C(t)为正值: {all(c > 0 for c in c_t_values)}")
    print(f"✓ C(t)值合理性: {all(0 < c < 100 for c in c_t_values)}")  # 经验范围

    # 分析多义词的C(t)特征
    polysemous_tokens = ['打', '行', '好']  # 常见的多义词
    print(f"多义词C(t)分析:")
    for result in results:
        if any(poly in result['token'] for poly in polysemous_tokens):
            print(f"  '{result['token']}': C(t)={result['c_t']:.6f}, M(t)={result['m_t']:.6f}")

    print()
    print("测试完成！")
    return results

if __name__ == "__main__":
    test_c_t_calculation()
