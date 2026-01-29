"""
测试句子"江河湖海都是水"的S(t)计算
输出每个token的最终S(t)值到CSV
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import pandas as pd
from s_t_calculator import SemanticDriftCoeff

def test_sentence():
    """测试句子并输出S(t)值"""
    
    # 测试句子
    test_sentence = "江河湖海都是水"
    print(f"测试句子: {test_sentence}")
    print("="*80)
    
    # 初始化计算器（模式3：全局锚点+L2归一化）
    calculator = SemanticDriftCoeff(
        lambda_decay=0.1,
        middle_layer_idx=12,
        normalize_hidden=True,      # L2归一化
        use_global_anchor=True      # 全局锚点（模式3）
    )
    
    print("\n配置:")
    print(f"  模式: 全局锚点（模式3）")
    print(f"  L2归一化: 开启")
    print(f"  参数: λ=0.1, μ=0.25, ω=0.3, ν=0.2")
    print(f"  LLM: Qwen2.5-0.5B-Instruct, Layer 12")
    print("\n计算中...")
    
    # 计算S(t)
    S_t = calculator(test_sentence)
    
    # 获取tokens
    from llm_hidden_extractor import extract_hidden_states
    model_name = "D:\\liubotao\\other\\BIT_TS\\LLM_GCG\\code\\models\\Qwen2___5-0___5B-Instruct"
    _, _, tokenizer, inputs, _ = extract_hidden_states(
        text=test_sentence,
        model_name=model_name,
        middle_layer_idx=12,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 解码tokens
    tokens = []
    for i in range(len(inputs['input_ids'][0])):
        token_text = tokenizer.decode([inputs['input_ids'][0][i]])
        tokens.append(token_text)
    
    # 准备数据
    results = []
    print("\n" + "="*80)
    print("【S(t) 最终结果】")
    print("="*80)
    print(f"\n{'Token':<8} {'位置':<6} {'S(t)':<10}")
    print("-"*80)
    
    for i, (token, s_t_value) in enumerate(zip(tokens, S_t)):
        s_t_val = s_t_value.item()
        print(f"{token:<8} {i:<6} {s_t_val:.6f}")
        
        results.append({
            'Token': token,
            '位置': i,
            'S(t)': s_t_val
        })
    
    # 保存到CSV
    df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(__file__), 'S_t_final_results.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*80)
    print(f"结果已保存到: {output_path}")
    print("="*80)
    
    # 统计信息
    s_t_values = [r['S(t)'] for r in results]
    print(f"\n统计信息:")
    print(f"  平均值: {sum(s_t_values)/len(s_t_values):.6f}")
    print(f"  最小值: {min(s_t_values):.6f} (Token: {tokens[s_t_values.index(min(s_t_values))]})")
    print(f"  最大值: {max(s_t_values):.6f} (Token: {tokens[s_t_values.index(max(s_t_values))]})")
    print(f"  标准差: {(sum((x-sum(s_t_values)/len(s_t_values))**2 for x in s_t_values)/len(s_t_values))**0.5:.6f}")
    
    print("\n" + "="*80)
    print("解释:")
    print("  S(t) → 0: 与全局主题高度一致，语义稳定")
    print("  S(t) → 1: 偏离全局主题，语义漂移显著")
    print("="*80)

if __name__ == "__main__":
    test_sentence()
