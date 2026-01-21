import os
import sys
import csv
import numpy as np
import pandas as pd
import torch

# Set up paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from k_t_calculator import compute_local_sectional_curvature
from d_t_calculator import compute_d_t
from s_t_calculator import SemanticDriftCalculator, GlobalSemanticAnchor
from m_t_calculator import compute_m_t_full
from llm_hidden_extractor import extract_hidden_states

# Example sentences (Chinese characters)
sentences = [
     "江河湖海都是水",
    "江边有很多人钓鱼。",
    "河水涨了，桥快被淹没。",
    "湖里有几只鸭子。",
    "海风很大，浪也高。",
    "池中种了荷花。",
    "泉水可以直接饮用。",
    "雨下了一整天。",
    "浪拍打着岸边。",
    "波纹在水面扩散。",
    "溪水很清澈。",
    "沙地上很难走路。",
    "山上有很多松树。",
    "石头很重，搬不动。",
    "岛上没有人居住。",
    "岩壁上有很多裂缝。",
    "洞里很黑，需要手电筒。",
    "泥巴弄脏了鞋子。",
    "林间有很多鸟叫声。",
    "草地上有露水。",
    "花开得很鲜艳。",
    "树下可以乘凉。",
    "竹子长得很快。",
    "松针掉了一地。",
    "梅树已经结果了。",
    "荷叶上有水珠。",
    "牛在田里吃草。",
    "羊喜欢在山坡上活动。",
    "鱼在水里游来游去。",
    "鸟在树枝上唱歌。",
    "马在草原上奔跑。",
    "虎在林中休息。",
    "狗在院子里晒太阳。",
    "猫在窗台上打盹。",
    "云很低，快要下雨了。",
    "风把树叶吹落了。",
    "雪覆盖了整个院子。",
    "雷声很响，小孩有点害怕。",
    "雾让路变得模糊。",
    "金项链很贵。",
    "木桌子很结实。",
    "铁门很重。",
    "铜壶可以煮水。",
    "银勺子很亮。",
    "火把木头烧成灰。",
    "电灯突然灭了。",
]

# Layers to process
layers = list(range(0, 24))  # 0 to 23 (Qwen2.5-0.5B has 24 layers)

# Initialize model and tokenizer once
_, _, tokenizer, _, _ = extract_hidden_states(sentences[0], middle_layer_idx=layers[0])
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "D:\\liubotao\\other\\BIT_TS\\LLM_GCG\\code\\models\\Qwen2___5-0___5B-Instruct",
    trust_remote_code=True,
    device_map="cuda" if torch.cuda.is_available() else "cpu"
).eval()

# Initialize S(t) calculator
s_calculator = SemanticDriftCalculator(model=model, tokenizer=tokenizer)

# Prepare data for CSV
data = []

# For each sentence, process all layers
for sentence in sentences:
    tokens = None  # Initialize tokens per sentence
    prev_k = None  # Reset prev_k per sentence
    prev_d = None  # Reset prev_d per sentence
    for layer in layers:
        hidden_states, token_num, tokenizer, inputs, attentions = extract_hidden_states(sentence, middle_layer_idx=layer)
        
        # Get token texts (only once per sentence, since same for all layers of the same sentence)
        if tokens is None:
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Compute S(t) for all tokens in this sentence and layer
        s_t_array = s_calculator.compute_s_t_batch(hidden_states, tokens, attentions)
        
        for token_idx in range(token_num):
            h_t = hidden_states[token_idx]
            token_text = tokenizer.decode([inputs['input_ids'][0][token_idx]])  # Decode to readable text
            
            # Calculate M(t)
            m_t = compute_m_t_full(h_t, token_text, tokens, token_idx, hidden_states)
            
            # Calculate K(t)
            k_t = compute_local_sectional_curvature(h_t, hidden_states, token_idx, token_num, precomputed_m_t=m_t)
            
            # Calculate D(t)
            d_t = compute_d_t(h_t, hidden_states, token_idx, token_num, precomputed_m_t=m_t)
            
            # Get S(t) from computed array
            s_t = s_t_array[token_idx]
            
            # Calculate dK/dt(L) = K(L) - K(L-1)
            dk_dt = k_t - prev_k if prev_k is not None else 0
                
            
            # Calculate |ΔD(L)| = |D(L) - D(L-1)|
            delta_d = abs(d_t - prev_d) if prev_d is not None else 0
            
            # Append to data
            data.append({
                '句子': sentence,
                '层': layer,
                'Token': token_text,
                '曲率 K(t)': k_t,
                '曲率演化率 dK/dt(L)': dk_dt,
                '平均欧氏距离 D(t)': d_t,
                '距离变化率 |ΔD(L)|': delta_d,
                '语义漂移系数 S(t)': s_t
            })
            
            # Update prev for next token/layer
            prev_k = k_t
            prev_d = d_t

# Write to CSV
start_layer = layers[0]
end_layer = layers[-1]
csv_file = os.path.join(ROOT, 'sd', 'feedback', f'layers_{start_layer}_{end_layer}_batch_core_variables.csv')
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['句子', '层', 'Token', '曲率 K(t)', '曲率演化率 dK/dt(L)', '平均欧氏距离 D(t)', '距离变化率 |ΔD(L)|', '语义漂移系数 S(t)'])
    writer.writeheader()
    writer.writerows(data)

print(f"Batch computation completed. Results saved to {csv_file}")