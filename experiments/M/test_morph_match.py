import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from m_t_calculator import compute_m_t_full
from llm_hidden_extractor import extract_hidden_states

# 测试文本
text = "江河湖海都是水"

# 提取隐藏状态
hidden_states, token_num, tokenizer, inputs, attn_weights = extract_hidden_states(
    text=text,
    middle_layer_idx=12  # 语义层
)

# 获取token文本
input_ids = inputs['input_ids'].squeeze(0)
tokens = [tokenizer.decode([token_id]) for token_id in input_ids]

print(f"文本: {text}")
print(f"Token序列: {tokens}")

# 依次输出每个字的M(t)
for idx, (token, h_t) in enumerate(zip(tokens, hidden_states)):
    if '\u4e00' <= token <= '\u9fff':  # 只输出中文汉字
        m_t = compute_m_t_full(
            h_t=h_t,
            token_text=token,
            tokens=tokens,
            token_idx=idx,
            hidden_states=hidden_states,
            model=None,
            tokenizer=None,
            attention_weights=attn_weights,
            beta=0.2,
            layer_idx=12
        )
        print(f"Token: '{token}' (位置: {idx}) -> M(t): {m_t:.6f}")
