# 预测指定层数的指标并与实际数据对比，生成图表
# 复用coupling_adjust.py中的校准系数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ====================== 指定预测层数范围 ======================
PREDICT_START_LAYER = 10  # 预测开始层
PREDICT_END_LAYER = 16   # 预测结束层

# Set up paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ====================== 从文件读取已校准的系数 ======================
import json
coeff_file = os.path.join(ROOT, 'sd', 'feedback', 'coupling_coefficients.json')
try:
    with open(coeff_file, 'r') as f:
        coefficients = json.load(f)
    ALPHA = coefficients['ALPHA']
    BETA = coefficients['BETA']
    OMEGA = coefficients['OMEGA']
    print(f"成功加载系数：α={ALPHA:.4f}, β={BETA:.4f}, ω={OMEGA:.4f}")
except FileNotFoundError:
    print(f"警告：系数文件 {coeff_file} 不存在，请先运行 coupling_adjust.py 进行系数校准")
    exit(1)

# Set up paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ====================== 读取数据 ======================
csv_path = os.path.join(ROOT, 'sd', 'feedback', 'layers_0_23_batch_core_variables.csv')
df = pd.read_csv(csv_path)

# 过滤指定句子
sentence = '羊喜欢在山坡上活动。'
df_sentence = df[df['句子'] == sentence]

# 获取该句子的所有unique tokens
tokens = df_sentence['Token'].unique()

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 预测函数 ======================
def predict_token_layers(token, start_layer, end_layer, alpha, beta, omega, token_df):
    """
    为单个token预测指定层范围的指标，从第9层开始迭代预测
    使用简化公式：dK/dt(L) = α·dK/dt(L-1) - β·(S(L-1)^2 + ω·|ΔD(L-1)|)
    """
    # 按层聚合该token的数据
    layer_data_token = token_df.groupby('层').agg({
        '曲率 K(t)': 'mean',
        '平均欧氏距离 D(t)': 'mean',
        '语义漂移系数 S(t)': 'mean',
        '曲率演化率 dK/dt(L)': lambda x: pd.to_numeric(x.replace('-', np.nan), errors='coerce').mean(),
        '距离变化率 |ΔD(L)|': lambda x: pd.to_numeric(x.replace('-', np.nan), errors='coerce').mean()
    }).reset_index()

    if len(layer_data_token) < 9:  # 需要至少到第9层
        return pd.DataFrame()

    layers_token = layer_data_token['层'].values
    K_token = layer_data_token['曲率 K(t)'].values
    D_token = layer_data_token['平均欧氏距离 D(t)'].values
    S_token = layer_data_token['语义漂移系数 S(t)'].values

    # 计算dKdt和abs_delta_D
    dKdt_token = np.diff(K_token)
    delta_D_token = np.diff(D_token)
    abs_delta_D_token = np.abs(delta_D_token)

    # 拟合D和S的层间趋势（使用1-9层的数据）
    mask = layers_token <= 9
    layers_fit = layers_token[mask]
    D_fit = D_token[mask]
    S_fit = S_token[mask]

    if len(layers_fit) > 1:
        a_D, b_D = np.polyfit(layers_fit, D_fit, deg=1)
        a_S, b_S = np.polyfit(layers_fit, S_fit, deg=1)
    else:
        a_D, b_D = 0, D_fit[0] if len(D_fit) > 0 else 0
        a_S, b_S = 0, S_fit[0] if len(S_fit) > 0 else 0

    # 初始化预测数据（从第9层开始）
    predicted_data = []

    # 第9层的已知值 - 找到第9层的索引
    layer9_idx = np.where(layers_token == 9)[0]
    if len(layer9_idx) > 0:
        idx9 = layer9_idx[0]
        current_K = K_token[idx9]  # 第9层K
        current_D = D_token[idx9]
        current_S = S_token[idx9]
        # 初始化前一个dKdt：使用第8-9层间的变化
        prev_dKdt = dKdt_token[idx9-1] if idx9 > 0 else 0  # 第8-9层间的变化
    else:
        # 如果没有第9层数据，用第8层数据
        idx8 = np.where(layers_token == 8)[0][0]
        current_K = K_token[idx8]
        current_D = D_token[idx8]
        current_S = S_token[idx8]
        prev_dKdt = 0

    for layer in range(start_layer, end_layer + 1):  # 预测指定层
        # 预测第L层时，使用第L-1层的指标作为输入
        # 检查第L-1层是否有实际观测数据
        prev_layer = layer - 1
        prev_layer_idx = np.where(layers_token == prev_layer)[0]
        if len(prev_layer_idx) > 0:
            # 如果有实际数据，使用第L-1层的实际S和D值
            idx = prev_layer_idx[0]
            S_prev = S_token[idx]
            D_prev = D_token[idx]
        else:
            # 如果没有实际数据，用趋势推断第L-1层的S和D
            S_prev = a_S * prev_layer + b_S
            D_prev = a_D * prev_layer + b_D

        # 计算|ΔD(L-1)|：第L-1层相对于第L-2层的距离变化
        delta_D_prev = D_prev - current_D
        abs_delta_D_prev = abs(delta_D_prev)

        # 用新的简化耦合反馈方程预测当前层(L)的曲率演化速率
        # dK/dt(L) = const + α·(-dK/dt(L-1)) - β·(S(L-1)^2 + ω·|ΔD(L-1)|)
        # 其中dK/dt(L-1)是第L-2到L-1层间的变化
        dKdt_pred = alpha * (-prev_dKdt) - beta * (S_prev ** 2 + omega * abs_delta_D_prev)

        # 预测当前层的曲率
        K_pred = current_K + dKdt_pred

        # 添加到预测数据
        predicted_data.append({
            '层': layer,
            '预测曲率 K(t)': K_pred,
            '预测曲率演化率 dK/dt(L)': dKdt_pred
        })

        # 更新当前值（为预测下一层做准备）
        # 当前值现在变成第L层的预测值
        current_K = K_pred
        prev_dKdt = dKdt_pred  # 预测的dKdt将成为下一层的prev_dKdt
        current_D = D_prev  # 第L-1层的D
        current_S = S_prev  # 第L-1层的S

    return pd.DataFrame(predicted_data)

# ====================== 为所有token生成对比图表 ======================
# 收集所有token的预测和实际数据
all_data = []
for token in tokens:
    token_df = df_sentence[df_sentence['Token'] == token]

    # 获取实际数据
    actual_data = token_df[(token_df['层'] >= PREDICT_START_LAYER) & (token_df['层'] <= PREDICT_END_LAYER)]

    # 预测数据
    predicted_df = predict_token_layers(token, PREDICT_START_LAYER, PREDICT_END_LAYER, ALPHA, BETA, OMEGA, token_df)

    if predicted_df.empty or actual_data.empty:
        continue

    # 合并数据用于绘图
    merged = pd.merge(actual_data[['层', '曲率 K(t)', '曲率演化率 dK/dt(L)']], predicted_df, on='层')
    all_data.append({'token': token, 'data': merged})

# 计算子图布局
num_tokens = len(all_data)
num_cols = 2  # 一行两个图
num_rows = (num_tokens + 1) // 2  # 计算需要的行数

# 生成曲率K(t)对比图
fig_k, axes_k = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
fig_k.suptitle(f'{sentence} - 所有Token曲率K(t)预测vs实际对比 (层{PREDICT_START_LAYER}-{PREDICT_END_LAYER})', fontsize=16)

for i, item in enumerate(all_data):
    row = i // num_cols
    col = i % num_cols
    ax = axes_k[row, col] if num_rows > 1 else axes_k[col]

    token = item['token']
    data = item['data']

    ax.plot(data['层'], data['曲率 K(t)'], 'b-o', label='实际 K(t)', linewidth=2)
    ax.plot(data['层'], data['预测曲率 K(t)'], 'r--s', label='预测 K(t)', linewidth=2)
    ax.set_xlabel('层')
    ax.set_ylabel('曲率 K(t)')
    ax.set_title(f'Token: {token}')
    ax.legend()
    ax.grid(True)

# 隐藏多余的子图
if num_tokens % 2 != 0 and num_rows > 1:
    axes_k[-1, -1].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'sd', 'feedback', 'all_tokens_curvature_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 生成曲率变化率dK/dt(L)对比图
fig_dk, axes_dk = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
fig_dk.suptitle(f'{sentence} - 所有Token曲率变化率dK/dt(L)预测vs实际对比 (层{PREDICT_START_LAYER}-{PREDICT_END_LAYER})', fontsize=16)

for i, item in enumerate(all_data):
    row = i // num_cols
    col = i % num_cols
    ax = axes_dk[row, col] if num_rows > 1 else axes_dk[col]

    token = item['token']
    data = item['data']

    ax.plot(data['层'], data['曲率演化率 dK/dt(L)'], 'b-o', label='实际 dK/dt(L)', linewidth=2)
    ax.plot(data['层'], data['预测曲率演化率 dK/dt(L)'], 'r--s', label='预测 dK/dt(L)', linewidth=2)
    ax.set_xlabel('层')
    ax.set_ylabel('曲率演化率 dK/dt(L)')
    ax.set_title(f'Token: {token}')
    ax.legend()
    ax.grid(True)

# 隐藏多余的子图
if num_tokens % 2 != 0 and num_rows > 1:
    axes_dk[-1, -1].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'sd', 'feedback', 'all_tokens_curvature_rate_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"对比图表已生成，共{num_tokens}个token。")
print("曲率对比图保存为: all_tokens_curvature_comparison.png")
print("曲率变化率对比图保存为: all_tokens_curvature_rate_comparison.png")
