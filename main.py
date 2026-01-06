from llm_hidden_extractor import extract_hidden_states
from curvature_calculator import compute_local_sectional_curvature
import matplotlib.pyplot as plt
import numpy as np

def plot_curvature_by_layer(curvature_data, token_texts, layers, text):
    """
    绘制各层各token的曲率图
    
    Args:
        curvature_data: 曲率数据，shape=(num_layers, num_tokens)
        token_texts: 各token的文本列表
        layers: 层编号列表
        text: 原始输入文本
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 科研风格配色：使用专业配色方案
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(layers)))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    num_tokens = len(token_texts)
    x = np.arange(num_tokens)
    width = 0.11  # 每个柱子的宽度
    
    # 为每一层绘制柱状图
    for i, layer in enumerate(layers):
        offset = (i - len(layers)/2) * width
        bars = ax.bar(x + offset, curvature_data[i], width, 
                     label=f'Layer {layer}', 
                     color=colors[i],
                     edgecolor='white',
                     linewidth=1.2,
                     alpha=0.85)
    
    ax.set_xlabel('Token Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sectional Curvature K(t)', fontsize=13, fontweight='bold')
    ax.set_title(f'Local Sectional Curvature Distribution\nText: "{text}"', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}\n{token}' for i, token in enumerate(token_texts)], 
                       fontsize=10)
    
    # 科研风格图例
    ax.legend(loc='upper right', frameon=True, fancybox=False, 
             shadow=False, framealpha=0.95, edgecolor='gray')
    
    # 网格样式
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)
    
    # 边框样式
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')
    
    plt.tight_layout()
    plt.savefig('curvature_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("\n图表已保存为 curvature_plot.png")

def main():
    # 1. 输入文本
    text = "他打了补丁"
    
    # 存储所有层的曲率数据
    all_curvatures = []
    layers = list(range(10, 17))  # 10到16层
    token_texts = []
    
    # 2. 循环提取10~16层的曲率
    for middle_layer in layers:
        print(f"\n{'='*60}")
        print(f"中间层编号：{middle_layer}")
        print(f"{'='*60}")
        
        hidden_states, token_num, tokenizer, inputs = extract_hidden_states(
            text=text,
            middle_layer_idx=middle_layer
        )
        print(f"文本token数：{token_num}")
        
        # 3. 计算每个token的局部截面曲率K(t)
        print("\n各token的局部截面曲率：")
        layer_curvatures = []
        
        for token_idx in range(token_num):
            h_t = hidden_states[token_idx]
            K_t = compute_local_sectional_curvature(
                h_t=h_t,
                hidden_states=hidden_states,
                token_idx=token_idx,
                sentence_length=token_num
            )
            layer_curvatures.append(K_t)
            
            # 第一层时收集token文本
            if middle_layer == layers[0]:
                token_id = inputs["input_ids"][0][token_idx].item()
                token_text = tokenizer.decode([token_id])
                token_texts.append(token_text)
            
            # 输出token内容和对应的K(t)
            token_id = inputs["input_ids"][0][token_idx].item()
            token_text = tokenizer.decode([token_id])
            print(f"Token[{token_idx}]：{token_text}")
            print(f"  h(t) 形状：{h_t.shape}")
            print(f"  h(t) 前10维：{h_t.numpy()[:10]}")
            print(f"  K(t) = {K_t:.6e}")
            print(f"  K(t) 十进制 = {K_t:.10f}")
            print(f"  K(t) 字符串 = '{str(K_t)}'\n")
        
        all_curvatures.append(layer_curvatures)
    
    # 4. 绘制曲率图
    curvature_array = np.array(all_curvatures)
    plot_curvature_by_layer(curvature_array, token_texts, layers, text)

if __name__ == "__main__":
    main()