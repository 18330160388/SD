import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_hidden_states(
    text: str,
    model_name: str = "D:\\liubotao\\other\\BIT_TS\\LLM_GCG\\code\\models\\Qwen2___5-0___5B-Instruct",
    middle_layer_idx: int = 12,  # 默认第12层，支持变量传入
    device: str = None
) -> tuple[torch.Tensor, int, AutoTokenizer, dict, torch.Tensor]:
    """
    提取千问7B指定中间层的语义状态向量h(t)
    
    Args:
        text: 输入中文文本
        model_name: 模型名称/本地路径
        middle_layer_idx: 中间语义层编号（0~23，千问7B共24层）
        device: 运行设备（自动识别GPU/CPU）
    
    Returns:
        hidden_states: 语义状态向量h(t)，shape=(token_num, hidden_dim)
        token_num: 文本的token数量
        tokenizer: tokenizer对象
        inputs: tokenizer编码后的输入
        attentions: 指定层的注意力权重，shape=(num_heads, token_num, token_num)
    """
    # 自动选择设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=device
    ).eval()  # 推理模式，禁用Dropout
    
    # 文本编码
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=512
    ).to(device)
    
    # 提取所有层的隐藏状态和注意力权重
    with torch.no_grad():  # 禁用梯度计算，加速并节省显存
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
    
    # 获取指定中间层的h(t)（shape: (batch_size, token_num, hidden_dim)）
    middle_hidden = outputs.hidden_states[middle_layer_idx]
    # 去除batch维度，返回(token_num, hidden_dim)
    h_t = middle_hidden.squeeze(0).cpu()  # 移到CPU便于后续计算
    token_num = h_t.shape[0]
    
    # 获取指定层的注意力权重（shape: (batch_size, num_heads, token_num, token_num)）
    middle_attentions = outputs.attentions[middle_layer_idx]
    # 对所有注意力头取平均，得到 (token_num, token_num)
    # avg_attentions[i, j] 表示 token i 对 token j 的注意力
    avg_attentions = middle_attentions.mean(dim=1).squeeze(0).cpu()
    # 计算每个token对全局的贡献度：所有其他tokens对它的注意力之和
    # attn_weights[j] = Σ_i avg_attentions[i, j]（所有query对token j的注意力）
    attn_weights = avg_attentions.sum(dim=0)  # shape: (token_num,)
    
    # 将inputs也移到CPU，避免后续使用时设备不匹配
    inputs = {k: v.cpu() for k, v in inputs.items()}
    
    return h_t, token_num, tokenizer, inputs, attn_weights

# 测试代码（单独运行时验证）
if __name__ == "__main__":
    test_text = "他打了补丁"
    h_t, token_num, tokenizer, inputs = extract_hidden_states(
        text=test_text,
        middle_layer_idx=12  # 可修改为10~16之间的任意层
    )
    print(f"输入文本：{test_text}")
    print(f"Token数量：{token_num}")
    print(f"h(t)形状：{h_t.shape}")  # 输出示例：torch.Size([5, 896])（Qwen2.5-0.5B）