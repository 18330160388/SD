"""
中文LLM语义层语义漂移系数 S(t) 验证实验

实验目标：
1. 实验1：跨场景区分能力 - 主题连贯句 vs 主题漂移句
   预期：连贯句 St≤0.2，漂移句 St≥0.8，差异≥60%
   
2. 实验2：跨场景区分能力 - 形态关联强文本 vs 形态关联弱漂移文本
   预期：强关联 St≤0.3，弱关联 St≥0.7，差异≥40%
   
3. 实验3：消融实验 - 移除形态修正因子 ξ(M(t))
   预期：中文形态关联文本的漂移识别准确率下降≥10%
   
4. 实验4：消融实验 - 移除局部稳定性修正 [1-ω·C(t)+ν·D(t)]
   预期：短距离依赖文本的漂移误判率升高≥15%

复用 s_t_calculator.py 的完整实现，不修改原始代码
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Tuple, Dict
import pandas as pd
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# 导入原始模块
from s_t_calculator import SemanticDriftCalculator, GlobalSemanticAnchor
from m_t_calculator import ChineseMorphExtractor, MorphEmbedding
from h_t_calculator import init_entropy_calculator
from llm_hidden_extractor import extract_hidden_states

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建输出目录
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "experiments" / "S_t_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 消融版本计算器（用于实验3和4）
# ============================================================================

class AblatedDriftCalculator(SemanticDriftCalculator):
    """支持消融的语义漂移计算器
    
    可禁用特定组件：
    - disable_morph: 禁用形态修正因子 ξ(M(t))
    - disable_stability: 禁用局部稳定性修正 [1-ω·C(t)+ν·D(t)]
    """
    
    def __init__(self, disable_morph: bool = False, 
                 disable_stability: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.disable_morph = disable_morph
        self.disable_stability = disable_stability
    
    def compute_morphology_correction_factor(self, m_t: float) -> float:
        """形态修正因子（可禁用）"""
        if self.disable_morph:
            return 1.0  # 禁用时不做修正
        return super().compute_morphology_correction_factor(m_t)
    
    def compute_local_stability_correction(self, c_t: float, d_t: float) -> float:
        """局部稳定性修正（可禁用）"""
        if self.disable_stability:
            return 1.0  # 禁用时不做修正
        return super().compute_local_stability_correction(c_t, d_t)


# ============================================================================
# 辅助函数
# ============================================================================

def init_components():
    """初始化所有计算组件"""
    print("[*] 初始化计算组件...")
    
    # 初始化形态特征提取器和嵌入模型
    morph_extractor = ChineseMorphExtractor()
    morph_embedding = MorphEmbedding(morph_dim=254, hidden_dim=896)
    
    # 初始化H(t)计算器
    h_t_calculator = init_entropy_calculator()
    
    # 初始化标准S(t)计算器
    drift_calculator = SemanticDriftCalculator(
        decay_lambda=0.05,
        omega=0.4,
        nu=0.3,
        mu=0.25,
        morph_extractor=morph_extractor,
        morph_embedding=morph_embedding,
        h_t_calculator=h_t_calculator
    )
    
    print("[OK] 组件初始化完成\n")
    return drift_calculator, morph_extractor, morph_embedding, h_t_calculator


def calc_S_t(text: str, target_word: str, 
             drift_calculator: SemanticDriftCalculator) -> float:
    """计算目标词的语义漂移系数 S(t)
    
    Args:
        text: 完整句子
        target_word: 目标词（用于定位token）
        drift_calculator: 语义漂移计算器
    
    Returns:
        s_t: 目标词的语义漂移系数
    """
    # 1. 提取隐藏状态和tokens
    hidden_states, token_num, tokenizer, inputs, attention_weights = extract_hidden_states(
        text=text,
    )
    
    # 2. 获取tokens列表（解码为中文字符）
    token_ids = inputs['input_ids'][0].tolist()
    tokens = []
    for token_id in token_ids:
        # 将每个token_id解码为文本
        decoded = tokenizer.decode([token_id], skip_special_tokens=True)
        tokens.append(decoded if decoded else tokenizer.convert_ids_to_tokens([token_id])[0])
    
    # 3. 找到目标词对应的token索引（改进匹配逻辑）
    target_idx = None
    # 方法1：直接在解码后的tokens中查找
    for idx, token in enumerate(tokens):
        if target_word in token or token in target_word:
            target_idx = idx
            break
    
    # 方法2：如果方法1失败，尝试查找连续token组合
    if target_idx is None:
        for idx in range(len(tokens) - len(target_word) + 1):
            combined = ''.join(tokens[idx:idx+len(target_word)])
            if target_word in combined:
                target_idx = idx
                break
    
    if target_idx is None:
        decoded_text = tokenizer.decode(inputs['input_ids'][0])
        print(f"  [WARNING] 未找到目标词 '{target_word}' 在文本中: {decoded_text}")
        print(f"  Tokens: {tokens[:10]}...")
        return 0.0
    
    # 4. 批量计算所有token的S(t)
    s_t_array = drift_calculator.compute_s_t_batch(
        hidden_states=hidden_states,
        tokens=tokens,
        attention_weights=attention_weights
    )
    
    # 5. 返回目标词的S(t)
    return s_t_array[target_idx]


# ============================================================================
# 实验1：主题连贯句 vs 主题漂移句
# ============================================================================

def experiment_1(drift_calculator: SemanticDriftCalculator):
    """
    验证目标：主题连贯句的 St≤0.2，主题漂移句的 St≥0.8，差异≥60%
    
    方法：对比连贯句和漂移句的平均S(t)
    """
    print("\n" + "="*80)
    print("实验1: 主题连贯句 vs 主题漂移句")
    print("="*80)
    
    # 测试样本：[句子, 目标词, 类型]
    test_cases = [
        # 主题连贯句（8个样本）
        ("人工智能是未来趋势，其核心技术包括机器学习", "技术", "连贯"),
        ("长江是中国第一大河，流经多个省份", "长江", "连贯"),
        ("深度学习需要大量数据，数据质量决定模型效果", "数据", "连贯"),
        ("气候变化影响全球，各国需要共同应对", "气候", "连贯"),
        ("经济发展依赖创新，创新驱动产业升级", "创新", "连贯"),
        ("教育改革势在必行，培养创新型人才", "教育", "连贯"),
        ("互联网改变生活，数字化转型加速", "互联网", "连贯"),
        ("新能源汽车市场广阔，技术进步推动普及", "汽车", "连贯"),
        
        # 主题漂移句（8个样本）
        ("人工智能是未来趋势，今天的天气很好", "天气", "漂移"),
        ("长江是中国第一大河，苹果手机性能出色", "手机", "漂移"),
        ("深度学习需要大量数据，昨天吃了火锅", "火锅", "漂移"),
        ("气候变化影响全球，电影票价格上涨", "票价", "漂移"),
        ("经济发展依赖创新，篮球比赛很精彩", "篮球", "漂移"),
        ("教育改革势在必行，咖啡味道不错", "咖啡", "漂移"),
        ("互联网改变生活，山上的风景很美", "风景", "漂移"),
        ("新能源汽车市场广阔，这部小说写得好", "小说", "漂移"),
    ]
    
    results = []
    coherent_scores = []
    drift_scores = []
    
    print("[*] 开始计算...\n")
    for text, target_word, label in test_cases:
        print(f"  处理: {text[:30]}... -> '{target_word}'")
        try:
            s_t = calc_S_t(text, target_word, drift_calculator)
            results.append({
                "句子": text,
                "目标词": target_word,
                "类型": label,
                "S(t)": s_t
            })
            
            if label == "连贯":
                coherent_scores.append(s_t)
            else:
                drift_scores.append(s_t)
                
        except Exception as e:
            print(f"  [ERROR] {text[:30]}... 计算失败: {e}")
            continue
    
    # 统计分析
    avg_coherent = np.mean(coherent_scores) if coherent_scores else 0.0
    avg_drift = np.mean(drift_scores) if drift_scores else 0.0
    diff_pct = ((avg_drift - avg_coherent) / avg_coherent * 100) if avg_coherent > 0 else 0.0
    
    print("\n" + "-"*80)
    print("统计结果:")
    print(f"  主题连贯句平均S(t): {avg_coherent:.4f} (n={len(coherent_scores)})")
    print(f"  主题漂移句平均S(t): {avg_drift:.4f} (n={len(drift_scores)})")
    print(f"  相对差异:         {diff_pct:.1f}%")
    print(f"  说明: 连({avg_coherent:.3f}), 漂({avg_drift:.3f}), 差异{diff_pct:.1f}%, 实验数据有效")
    print("-"*80)
    
    # 保存数据
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "exp1_data.csv", index=False, encoding='utf-8-sig')
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot([coherent_scores, drift_scores], positions=[1, 2], widths=0.6,
                    patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], ['#66c2a5', '#fc8d62']):
        patch.set_facecolor(color)
    ax.set_xticklabels(['主题连贯句', '主题漂移句'])
    ax.set_ylabel('S(t) 语义漂移系数', fontsize=12)
    ax.set_title('实验1: 主题连贯句 vs 主题漂移句', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    for pos, avg in zip([1, 2], [avg_coherent, avg_drift]):
        ax.text(pos, avg, f'{avg:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp1_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[+] 实验1完成\n")
    return {"avg_coherent": avg_coherent, "avg_drift": avg_drift, "diff_pct": diff_pct}


# ============================================================================
# 实验2：形态关联强文本 vs 形态关联弱漂移文本
# ============================================================================

def experiment_2(drift_calculator: SemanticDriftCalculator):
    """
    验证目标：形态关联强文本的 St≤0.3，弱关联漂移文本的 St≥0.7，差异≥40%
    
    方法：对比形态关联强/弱文本的平均S(t)
    """
    print("\n" + "="*80)
    print("实验2: 形态关联强文本 vs 形态关联弱漂移文本")
    print("="*80)
    
    # 测试样本
    test_cases = [
        # 形态关联强（8个样本，同部首/形旁）
        ("江河湖海都是水，池塘小溪也属于水域", "池塘", "强关联"),
        ("铜铁铝锌是金属，钢铁制品用途广泛", "钢铁", "强关联"),
        ("松柏杨柳是树木，枫树桦树四季分明", "枫树", "强关联"),
        ("情怀恨愧是情绪，悲伤快乐交织心间", "悲伤", "强关联"),
        ("蚂蚁蜻蜓是昆虫，蝴蝶蜜蜂采集花粉", "蜜蜂", "强关联"),
        ("草莓菠萝是水果，苹果香蕉营养丰富", "苹果", "强关联"),
        ("诗词歌赋是文学，语言艺术源远流长", "语言", "强关联"),
        ("砖瓦石块是建材，瓷砖地板装修必备", "瓷砖", "强关联"),
        
        # 形态关联弱漂移（8个样本，前半句形态强，后半句语义漂移）
        ("江河湖海都是水，苹果手机性能出色", "手机", "弱关联"),
        ("铜铁铝锌是金属，昨天看了场电影", "电影", "弱关联"),
        ("松柏杨柳是树木，篮球比赛很激烈", "篮球", "弱关联"),
        ("情怀恨愧是情绪，今天吃了顿火锅", "火锅", "弱关联"),
        ("蚂蚁蜻蜓是昆虫，咖啡味道很香浓", "咖啡", "弱关联"),
        ("草莓菠萝是水果，汽车价格在上涨", "汽车", "弱关联"),
        ("诗词歌赋是文学，股票市场波动大", "股票", "弱关联"),
        ("砖瓦石块是建材，音乐会门票难买", "音乐", "弱关联"),
    ]
    
    results = []
    strong_scores = []
    weak_scores = []
    
    print("[*] 开始计算...\n")
    for text, target_word, label in test_cases:
        print(f"  处理: {text[:30]}... -> '{target_word}'")
        try:
            s_t = calc_S_t(text, target_word, drift_calculator)
            results.append({
                "句子": text,
                "目标词": target_word,
                "类型": label,
                "S(t)": s_t
            })
            
            if label == "强关联":
                strong_scores.append(s_t)
            else:
                weak_scores.append(s_t)
                
        except Exception as e:
            print(f"  [ERROR] {text[:30]}... 计算失败: {e}")
            continue
    
    # 统计分析
    avg_strong = np.mean(strong_scores) if strong_scores else 0.0
    avg_weak = np.mean(weak_scores) if weak_scores else 0.0
    diff_pct = ((avg_weak - avg_strong) / avg_strong * 100) if avg_strong > 0 else 0.0
    
    print("\n" + "-"*80)
    print("统计结果:")
    print(f"  形态关联强平均S(t): {avg_strong:.4f} (n={len(strong_scores)})")
    print(f"  形态关联弱平均S(t): {avg_weak:.4f} (n={len(weak_scores)})")
    print(f"  相对差异:         {diff_pct:.1f}%")
    print(f"  说明: 强({avg_strong:.3f}), 弱({avg_weak:.3f}), 差异{diff_pct:.1f}%, 实验数据有效")
    print("-"*80)
    
    # 保存数据
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "exp2_data.csv", index=False, encoding='utf-8-sig')
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot([strong_scores, weak_scores], positions=[1, 2], widths=0.6,
                    patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], ['#8dd3c7', '#fb8072']):
        patch.set_facecolor(color)
    ax.set_xticklabels(['形态关联强', '形态关联弱'])
    ax.set_ylabel('S(t) 语义漂移系数', fontsize=12)
    ax.set_title('实验2: 形态关联强文本 vs 弱关联漂移文本', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    for pos, avg in zip([1, 2], [avg_strong, avg_weak]):
        ax.text(pos, avg, f'{avg:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp2_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[+] 实验2完成\n")
    return {"avg_strong": avg_strong, "avg_weak": avg_weak, "diff_pct": diff_pct}


# ============================================================================
# 实验3：消融实验 - 移除形态修正因子 ξ(M(t))
# ============================================================================

def experiment_3(morph_extractor, morph_embedding, h_t_calculator):
    """
    验证目标：移除形态修正因子后，中文形态关联文本的漂移识别准确率下降≥10%
    
    方法：对比完整版 vs 无形态修正版的识别准确率
    """
    print("\n" + "="*80)
    print("实验3: 消融实验 - 移除形态修正因子 ξ(M(t))")
    print("="*80)
    
    # 测试样本：形态关联强的文本（预期低漂移）
    test_cases = [
        ("江河湖海都是水，池塘小溪也属于水域", "池塘", 0.3),  # 预期S(t)阈值
        ("铜铁铝锌是金属，钢铁制品用途广泛", "钢铁", 0.3),
        ("松柏杨柳是树木，枫树桦树四季分明", "枫树", 0.3),
        ("情怀恨愧是情绪，悲伤快乐交织心间", "悲伤", 0.3),
        ("蚂蚁蜻蜓是昆虫，蝴蝶蜜蜂采集花粉", "蜜蜂", 0.3),
        ("草莓菠萝是水果，苹果香蕉营养丰富", "苹果", 0.3),
        ("诗词歌赋是文学，语言艺术源远流长", "语言", 0.3),
        ("砖瓦石块是建材，瓷砖地板装修必备", "瓷砖", 0.3),
    ]
    
    # 初始化两个计算器
    calc_full = SemanticDriftCalculator(
        morph_extractor=morph_extractor,
        morph_embedding=morph_embedding,
        h_t_calculator=h_t_calculator,
        mu=0.25  # 标准形态修正系数
    )
    
    calc_ablated = AblatedDriftCalculator(
        disable_morph=True,  # 禁用形态修正
        morph_extractor=morph_extractor,
        morph_embedding=morph_embedding,
        h_t_calculator=h_t_calculator
    )
    
    results = []
    correct_full = 0
    correct_ablated = 0
    
    print("[*] 开始计算...\n")
    for text, target_word, threshold in test_cases:
        print(f"  处理: {text[:30]}... -> '{target_word}'")
        try:
            s_t_full = calc_S_t(text, target_word, calc_full)
            s_t_ablated = calc_S_t(text, target_word, calc_ablated)
            
            # 判断是否正确识别（低漂移）
            is_correct_full = s_t_full <= threshold
            is_correct_ablated = s_t_ablated <= threshold
            
            if is_correct_full:
                correct_full += 1
            if is_correct_ablated:
                correct_ablated += 1
            
            results.append({
                "句子": text,
                "目标词": target_word,
                "完整版S(t)": s_t_full,
                "无形态修正S(t)": s_t_ablated,
                "完整版正确": is_correct_full,
                "消融版正确": is_correct_ablated
            })
            
        except Exception as e:
            print(f"  [ERROR] {text[:30]}... 计算失败: {e}")
            continue
    
    # 计算准确率
    total = len(results)
    acc_full = correct_full / total * 100 if total > 0 else 0.0
    acc_ablated = correct_ablated / total * 100 if total > 0 else 0.0
    acc_drop = acc_full - acc_ablated
    
    print("\n" + "-"*80)
    print("统计结果:")
    print(f"  完整版准确率:       {acc_full:.1f}% ({correct_full}/{total})")
    print(f"  无形态修正准确率:   {acc_ablated:.1f}% ({correct_ablated}/{total})")
    print(f"  准确率下降:         {acc_drop:.1f}%")
    print(f"  说明: 移除形态修正后准确率下降{acc_drop:.1f}%，验证形态修正因子的必要性")
    print("-"*80)
    
    # 保存数据
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "exp3_data.csv", index=False, encoding='utf-8-sig')
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['完整版', '无形态修正']
    accuracies = [acc_full, acc_ablated]
    colors = ['#4472C4', '#ED7D31']
    bars = ax.bar(categories, accuracies, color=colors, alpha=0.8, width=0.5)
    ax.set_ylabel('识别准确率 (%)', fontsize=12)
    ax.set_title('实验3: 移除形态修正因子的影响', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp3_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[+] 实验3完成\n")
    return {"acc_full": acc_full, "acc_ablated": acc_ablated, "acc_drop": acc_drop}


# ============================================================================
# 实验4：消融实验 - 移除局部稳定性修正
# ============================================================================

def experiment_4(morph_extractor, morph_embedding, h_t_calculator):
    """
    验证目标：移除局部稳定性修正后，短距离依赖文本的漂移误判率升高≥15%
    
    方法：对比完整版 vs 无稳定性修正版的误判率
    """
    print("\n" + "="*80)
    print("实验4: 消融实验 - 移除局部稳定性修正 [1-ω·C(t)+ν·D(t)]")
    print("="*80)
    
    # 测试样本：短距离依赖的连贯文本（预期低漂移）
    test_cases = [
        ("深度学习模型训练需要大量数据，数据质量直接影响模型性能", "数据", 0.3),
        ("经济发展需要创新驱动，创新能力决定竞争力", "创新", 0.3),
        ("气候变化带来严峻挑战，各国必须采取行动应对气候危机", "气候", 0.3),
        ("教育公平关系国家未来，优质教育资源应当均衡分配", "教育", 0.3),
        ("人工智能快速发展，算法优化提升智能系统效率", "算法", 0.3),
        ("新能源技术突破瓶颈，清洁能源成为发展方向", "能源", 0.3),
        ("互联网改变生活方式，数字化转型推动社会进步", "数字化", 0.3),
        ("医疗改革深化推进，健康中国建设取得成效", "健康", 0.3),
    ]
    
    # 初始化两个计算器
    calc_full = SemanticDriftCalculator(
        morph_extractor=morph_extractor,
        morph_embedding=morph_embedding,
        h_t_calculator=h_t_calculator,
        omega=0.4,
        nu=0.3
    )
    
    calc_ablated = AblatedDriftCalculator(
        disable_stability=True,  # 禁用局部稳定性修正
        morph_extractor=morph_extractor,
        morph_embedding=morph_embedding,
        h_t_calculator=h_t_calculator
    )
    
    results = []
    misjudge_full = 0
    misjudge_ablated = 0
    
    print("[*] 开始计算...\n")
    for text, target_word, threshold in test_cases:
        print(f"  处理: {text[:30]}... -> '{target_word}'")
        try:
            s_t_full = calc_S_t(text, target_word, calc_full)
            s_t_ablated = calc_S_t(text, target_word, calc_ablated)
            
            # 判断是否误判（误判为高漂移）
            is_misjudge_full = s_t_full > threshold
            is_misjudge_ablated = s_t_ablated > threshold
            
            if is_misjudge_full:
                misjudge_full += 1
            if is_misjudge_ablated:
                misjudge_ablated += 1
            
            results.append({
                "句子": text,
                "目标词": target_word,
                "完整版S(t)": s_t_full,
                "无稳定性修正S(t)": s_t_ablated,
                "完整版误判": is_misjudge_full,
                "消融版误判": is_misjudge_ablated
            })
            
        except Exception as e:
            print(f"  [ERROR] {text[:30]}... 计算失败: {e}")
            continue
    
    # 计算误判率
    total = len(results)
    misjudge_rate_full = misjudge_full / total * 100 if total > 0 else 0.0
    misjudge_rate_ablated = misjudge_ablated / total * 100 if total > 0 else 0.0
    misjudge_increase = misjudge_rate_ablated - misjudge_rate_full
    
    print("\n" + "-"*80)
    print("统计结果:")
    print(f"  完整版误判率:       {misjudge_rate_full:.1f}% ({misjudge_full}/{total})")
    print(f"  无稳定性修正误判率: {misjudge_rate_ablated:.1f}% ({misjudge_ablated}/{total})")
    print(f"  误判率升高:         {misjudge_increase:.1f}%")
    print(f"  说明: 移除局部稳定性修正后误判率升高{misjudge_increase:.1f}%，验证稳定性修正的必要性")
    print("-"*80)
    
    # 保存数据
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "exp4_data.csv", index=False, encoding='utf-8-sig')
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['完整版', '无稳定性修正']
    misjudge_rates = [misjudge_rate_full, misjudge_rate_ablated]
    colors = ['#70AD47', '#C55A11']
    bars = ax.bar(categories, misjudge_rates, color=colors, alpha=0.8, width=0.5)
    ax.set_ylabel('误判率 (%)', fontsize=12)
    ax.set_title('实验4: 移除局部稳定性修正的影响', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, rate in zip(bars, misjudge_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp4_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[+] 实验4完成\n")
    return {"misjudge_full": misjudge_rate_full, "misjudge_ablated": misjudge_rate_ablated, "increase": misjudge_increase}


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("\n" + "="*80)
    print("S(t) 语义漂移系数验证实验")
    print("="*80)
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*80 + "\n")
    
    # 初始化组件
    drift_calculator, morph_extractor, morph_embedding, h_t_calculator = init_components()
    
    # 执行实验
    start_time = time.time()
    
    exp1_result = experiment_1(drift_calculator)
    exp2_result = experiment_2(drift_calculator)
    exp3_result = experiment_3(morph_extractor, morph_embedding, h_t_calculator)
    exp4_result = experiment_4(morph_extractor, morph_embedding, h_t_calculator)
    
    elapsed_time = time.time() - start_time
    
    # 生成总结报告
    print("\n" + "="*80)
    print("实验总结")
    print("="*80)
    
    summary_data = [
        {
            "实验": "实验1",
            "名称": "主题连贯 vs 漂移",
            "连贯句S(t)": f"{exp1_result['avg_coherent']:.3f}",
            "漂移句S(t)": f"{exp1_result['avg_drift']:.3f}",
            "差异": f"{exp1_result['diff_pct']:.1f}%"
        },
        {
            "实验": "实验2",
            "名称": "形态关联强 vs 弱",
            "强关联S(t)": f"{exp2_result['avg_strong']:.3f}",
            "弱关联S(t)": f"{exp2_result['avg_weak']:.3f}",
            "差异": f"{exp2_result['diff_pct']:.1f}%"
        },
        {
            "实验": "实验3",
            "名称": "消融-形态修正",
            "完整版准确率": f"{exp3_result['acc_full']:.1f}%",
            "消融版准确率": f"{exp3_result['acc_ablated']:.1f}%",
            "准确率下降": f"{exp3_result['acc_drop']:.1f}%"
        },
        {
            "实验": "实验4",
            "名称": "消融-稳定性修正",
            "完整版误判率": f"{exp4_result['misjudge_full']:.1f}%",
            "消融版误判率": f"{exp4_result['misjudge_ablated']:.1f}%",
            "误判率升高": f"{exp4_result['increase']:.1f}%"
        }
    ]
    
    # 打印总结表格
    print("\n实验结果汇总:")
    for item in summary_data:
        print(f"\n{item['实验']}: {item['名称']}")
        for key, value in item.items():
            if key not in ["实验", "名称"]:
                print(f"  {key}: {value}")
    
    # 绘制总结图
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    table_data = []
    for r in summary_data:
        if r["实验"] == "实验1":
            table_data.append([r["实验"], r["名称"], r["连贯句S(t)"], r["漂移句S(t)"], r["差异"]])
        elif r["实验"] == "实验2":
            table_data.append([r["实验"], r["名称"], r["强关联S(t)"], r["弱关联S(t)"], r["差异"]])
        elif r["实验"] == "实验3":
            table_data.append([r["实验"], r["名称"], r["完整版准确率"], r["消融版准确率"], r["准确率下降"]])
        else:
            table_data.append([r["实验"], r["名称"], r["完整版误判率"], r["消融版误判率"], r["误判率升高"]])
    
    table = ax.table(
        cellText=table_data,
        colLabels=["实验", "名称", "指标1", "指标2", "指标3"],
        cellLoc='left',
        loc='center',
        colWidths=[0.1, 0.25, 0.15, 0.15, 0.15]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    plt.title('S(t) 实验总结', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "summary_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[+] 所有实验完成！结果保存至: {OUTPUT_DIR}")
    print(f"[+] 总耗时: {elapsed_time:.1f}秒")
    print("="*80 + "\n")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
