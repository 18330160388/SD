from llm_hidden_extractor import extract_hidden_states
from curvature_calculator import compute_local_sectional_curvature
from d_t_calculator import compute_d_t, compute_d_t_batch  # 新增导入
import matplotlib.pyplot as plt
import numpy as np

def plot_curvature_by_layer(curvature_data, token_texts, layers, text):
    """原绘图函数不变，此处省略（保持原有代码）"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(layers)))
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    num_tokens = len(token_texts)
    x = np.arange(num_tokens)
    width = 0.11
    
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
    
    ax.legend(loc='upper right', frameon=True, fancybox=False, 
             shadow=False, framealpha=0.95, edgecolor='gray')
    
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')
    
    plt.tight_layout()
    filename = f'curvature_plot_{text}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n曲率图表已保存为 {filename}")

# 新增D(t)绘图函数
def plot_d_t_by_layer(d_t_data, token_texts, layers, text):
    """绘制各层各token的D(t)分布"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(layers)))
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    num_tokens = len(token_texts)
    x = np.arange(num_tokens)
    width = 0.11
    
    for i, layer in enumerate(layers):
        offset = (i - len(layers)/2) * width
        bars = ax.bar(x + offset, d_t_data[i], width, 
                     label=f'Layer {layer}', 
                     color=colors[i],
                     edgecolor='white',
                     linewidth=1.2,
                     alpha=0.85)
    
    ax.set_xlabel('Token Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Semantic Distribution Distance D(t)', fontsize=13, fontweight='bold')
    ax.set_title(f'D(t) Distribution\nText: "{text}"', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}\n{token}' for i, token in enumerate(token_texts)], 
                       fontsize=10)
    
    ax.legend(loc='upper right', frameon=True, fancybox=False, 
             shadow=False, framealpha=0.95, edgecolor='gray')
    
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')
    
    plt.tight_layout()
    filename = f'd_t_plot_{text}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nD(t)图表已保存为 {filename}")

def main():
    text = "我用苹果砸开了手机"
    all_curvatures = []
    all_d_t = []  # 新增存储D(t)
    layers = list(range(10, 17))
    token_texts = []
    
    for middle_layer in layers:
        print(f"\n{'='*60}")
        print(f"中间层编号：{middle_layer}")
        print(f"{'='*60}")
        
        hidden_states, token_num, tokenizer, inputs = extract_hidden_states(
            text=text,
            middle_layer_idx=middle_layer
        )
        print(f"文本token数：{token_num}")
        
        # 计算曲率
        layer_curvatures = []
        # 计算D(t)
        layer_d_t = []
        
        for token_idx in range(token_num):
            h_t = hidden_states[token_idx]
            # 曲率计算（原有逻辑）
            K_t = compute_local_sectional_curvature(
                h_t=h_t,
                hidden_states=hidden_states,
                token_idx=token_idx,
                sentence_length=token_num
            )
            layer_curvatures.append(K_t)
            
            # D(t)计算（新增逻辑）
            D_t = compute_d_t(
                h_t=h_t,
                hidden_states=hidden_states,
                token_idx=token_idx,
                sentence_length=token_num,
                distance_type="euclidean",  # 可选：cosine
                normalize=True
            )
            layer_d_t.append(D_t)
            
            if middle_layer == layers[0]:
                token_id = inputs["input_ids"][0][token_idx].item()
                token_text = tokenizer.decode([token_id])
                token_texts.append(token_text)
            
            # 输出扩展：增加D(t)打印
            token_id = inputs["input_ids"][0][token_idx].item()
            token_text = tokenizer.decode([token_id])
            print(f"Token[{token_idx}]：{token_text}")
            print(f"  h(t) 形状：{h_t.shape}")
            print(f"  h(t) 前10维：{h_t.numpy()[:10]}")
            print(f"  K(t) = {K_t:.6e}")
            print(f"  D(t) = {D_t:.6f}")  # 新增D(t)输出
            print(f"  ------------------------")
        
        all_curvatures.append(layer_curvatures)
        all_d_t.append(layer_d_t)  # 保存当前层D(t)
    
    # 绘制曲率图（原有逻辑）
    curvature_array = np.array(all_curvatures)
    plot_curvature_by_layer(curvature_array, token_texts, layers, text)
    
    # 绘制D(t)图（新增）
    d_t_array = np.array(all_d_t)
    plot_d_t_by_layer(d_t_array, token_texts, layers, text)

if __name__ == "__main__":
    main()