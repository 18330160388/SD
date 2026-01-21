# 耦合反馈方程拟合完整代码（适配平均欧氏距离）
# 核心功能：1. 数据预处理 2. 多元线性回归拟合 3. 系数验证 4. 16层回测 5. 17层预测
import numpy as np
import statsmodels.api as sm
import pandas as pd
import os

# Set up paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ====================== 层数范围定义 ======================
START_LAYER = 2   # 开始层号 - 使用更大的样本范围
END_LAYER = 23    # 结束层号 - 覆盖更多层以增加样本量

# ====================== 耦合反馈方程系数定义 ======================
"""
耦合反馈方程简化形式（基于coupling.md的核心公式）：
dK/dt(L) = α·dK/dt(L-1) - β·(S(L-1)^2 + ω·|ΔD(L-1)|)

其中：
- dK/dt(L): 第L层的曲率演化速率（当前层曲率相对于前一层的变化率）
- dK/dt(L-1): 第L-1层的曲率演化速率（前一层曲率相对于前前层的变化率）
- S(L-1): 第L-1层的语义漂移系数（语义稳定性度量）
- |ΔD(L-1)|: 第L-1层的平均欧氏距离变化绝对值（距离变化的幅度）

物理意义与回路对应：
- α (正向驱动系数): 量化"上一层速率对当前层速率的正向驱动强度" - α>0证明"速率增加"环节成立
- β (总抑制系数): 统一放大"语义漂移平方+距离变化"的总抑制效果 - β>0证明整个耦合反馈回路是闭环的
- ω (距离权重系数): 量化"距离变化"在总抑制项中的相对权重 - ω>0证明距离变化对抑制有贡献

因果逻辑闭环：
dK/dt(L-1)↑ →[α] dK/dt(L)初始↑ →[语义演化] S(L)↑ →[S²] |ΔD(L)|↑ →[ω] 总抑制项↑ →[β] dK/dt(L)最终↓
"""

# 耦合反馈方程系数（每次校准后更新为最新值）
ALPHA = 0.5000    # 跨层驱动系数 α - 前层速率驱动强度
BETA = 0.3000     # 总抑制系数 β - 总抑制效果放大
OMEGA = 0.1500    # 距离权重系数 ω - 距离变化在抑制项中的权重

# 耦合反馈方程系数配置对象（每次校准后自动更新）
class CouplingConfig:
    def __init__(self):
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.OMEGA = OMEGA

# 创建全局配置对象
config = CouplingConfig()

# ====================== 第一步：从CSV读取并预处理数据 ======================
csv_path = os.path.join(ROOT, 'sd', 'feedback', 'layers_0_23_batch_core_variables.csv')
df = pd.read_csv(csv_path)

# 使用所有句子和所有Token的数据进行分析（聚合所有数据）
# df = df[df['句子'] == '江河湖海都是水']
# df = df[df['Token'] == '海']

# 按层分组，计算平均值（虽然每层只有一个海字，但保持一致性）
layer_data = df.groupby('层').agg({
    '曲率 K(t)': 'mean',
    '平均欧氏距离 D(t)': 'mean',
    '曲率演化率 dK/dt(L)': lambda x: pd.to_numeric(x.replace('-', np.nan), errors='coerce').mean(),
    '距离变化率 |ΔD(L)|': lambda x: pd.to_numeric(x.replace('-', np.nan), errors='coerce').mean(),
    '语义漂移系数 S(t)': 'mean'  # 新增：语义漂移系数
}).reset_index()

print("聚合后的层数据：")
print(layer_data)
print()

# 提取所有层的核心变量
layers = layer_data['层'].values
K = layer_data['曲率 K(t)'].values  # 曲率（所有层）
D = layer_data['平均欧氏距离 D(t)'].values  # 平均欧氏距离（所有层）
S = layer_data['语义漂移系数 S(t)'].values  # 语义漂移系数（所有层）

# ====================== 第二步：计算衍生变量 ======================
# 1. 曲率演化速率 dK/dt (2~M层，共M-1个值，其中M=len(layers))
dKdt = np.diff(K)  # [K2-K1, K3-K2, ..., KM-K(M-1)]

# 2. 平均欧氏距离变化速率 ΔD (2~M层，共M-1个值)
delta_D = np.diff(D)

# 3. 距离变化速率绝对值 |ΔD| (2~M层)
abs_delta_D = np.abs(delta_D)

# ====================== 第三步：提取拟合用数据 ======================
# 数据结构说明：
# layers: [1,2,3,...,M] (M个元素)
# K, D, S: [K1,K2,...,KM] (M个元素)
# dKdt = np.diff(K): [K2-K1, K3-K2, ..., KM-K(M-1)] (M-1个元素，对应2-M层)
# abs_delta_D = np.abs(np.diff(D)): 对应2-M层 (M-1个元素)

# 拟合目标：使用START_LAYER到END_LAYER范围的数据
# 新公式：dK/dt(L) = α·dK/dt(L-1) - β·(S(L-1)^2 + ω·|ΔD(L-1)|)
start_idx = START_LAYER - 1  # START_LAYER层的索引 (从0开始)
end_idx = END_LAYER - 1      # END_LAYER层的索引

# y: 预测START_LAYER+1到END_LAYER层的dKdt
y = dKdt[start_idx : end_idx]  # dKdt[START_LAYER-1 : END_LAYER-1]

# X1: 前1层dKdt (START_LAYER到END_LAYER-1层)
X1 = dKdt[start_idx-1 : end_idx-1]  # dK/dt(L-1)

# X2: 前1层S的平方 (START_LAYER到END_LAYER-1层)
X2 = S[start_idx : end_idx] ** 2  # S(L-1)²

# X3: 前1层|ΔD| (START_LAYER到END_LAYER-1层)
X3 = abs_delta_D[start_idx : end_idx]  # |ΔD(L-1)|

print(f"拟合层范围: {START_LAYER+1}-{END_LAYER}")
print(f"拟合样本数: {len(y)}")
print(f"X1 shape: {X1.shape}, X2 shape: {X2.shape}, X3 shape: {X3.shape}, y shape: {y.shape}")

# 构造回归矩阵（添加截距项）
X = np.column_stack((-X1, X2, X3))  # 注意：X1取负号来实现正反馈
X = sm.add_constant(X)  # 添加常数项

# ====================== 第四步：多元线性回归拟合 ======================
model = sm.OLS(y, X).fit()

# 提取拟合系数
# 新公式：dK/dt(L) = const + α·(-dK/dt(L-1)) - β·X2 - β·ω·X3
# 其中 -dK/dt(L-1)实现正反馈，X2 = S(L-1)², X3 = |ΔD(L-1)|
const = model.params[0]     # 常数项
alpha = model.params[1]     # 有效驱动系数 α (对应-X1，实现正反馈)
beta_s_squared = model.params[2]  # -β (对应X2 = S²)
beta_omega_delta_d = model.params[3]  # -β·ω (对应X3 = |ΔD|)

# 从回归系数推导β和ω
# 方法：标准化β=1，这样ω直接等于回归系数的负值除以β
beta = 1.0  # 标准化总抑制系数 β = 1
omega = -beta_omega_delta_d / beta if beta != 0 else 0  # 距离权重系数 ω

# 更新全局系数常量
ALPHA = alpha
BETA = beta
OMEGA = omega

# ====================== 第五步：输出拟合结果 ======================
print("="*50)
print("耦合反馈方程拟合结果")
print("="*50)
print(f"最终方程：dK/dt(L) = {const:.4f} + {alpha:.4f}·(-dK/dt(L-1)) - {BETA:.4f}·(S(L-1)² + {OMEGA:.4f}·|ΔD(L-1)|)")
print(f"拟合优度 R² = {model.rsquared:.4f}")
print(f"系数显著性（p值）：")
print(f"  常数项: p={model.pvalues[0]:.4f}")
print(f"  有效驱动系数α: p={model.pvalues[1]:.4f} (α>0表示正反馈机制)")
print(f"  语义平方抑制系数β: p={model.pvalues[2]:.4f} (负系数表示抑制)")
print(f"  距离变化抑制系数β·ω: p={model.pvalues[3]:.4f} (负系数表示抑制)")
print(f"推导参数：")
print(f"  总抑制系数β: {BETA:.4f}")
print(f"  距离权重系数ω: {OMEGA:.4f} (ω>0证明距离变化对抑制有贡献)")
print(f"权重关系：语义平方项权重=1.0, 距离变化项权重={OMEGA:.4f}")
print("\n完整回归报告：")
print(model.summary())

# ====================== 第六步：最后层回测验证 ======================
# 使用新的简化公式进行预测验证
# dK/dt(L) = const + α·(-dK/dt(L-1)) - β·(S(L-1)^2 + ω·|ΔD(L-1)|)
# 预测第END_LAYER层的dKdt，使用第(END_LAYER-1)层的数据
dKdt_l1 = dKdt[end_idx-1]       # dK/dt(L-1)
S_l1 = S[end_idx-1]             # S(L-1)
abs_delta_D_l1 = abs_delta_D[end_idx-1]  # |ΔD(L-1)|
S_l1_squared = S_l1 ** 2        # S(L-1)²

dKdt_last_pred = const + ALPHA * (-dKdt_l1) - BETA * (S_l1_squared + OMEGA * abs_delta_D_l1)
dKdt_last_true = dKdt[end_idx-1]    # 第END_LAYER层真实速率

# 计算回测误差
relative_error = abs(dKdt_last_pred - dKdt_last_true) / abs(dKdt_last_true) * 100 if dKdt_last_true != 0 else 0

print("="*50)
print(f"最后层（第{END_LAYER}层）回测验证")
print("="*50)
print(f"第{END_LAYER}层真实曲率速率：{dKdt_last_true:.4f}")
print(f"第{END_LAYER}层预测曲率速率：{dKdt_last_pred:.4f}")
print(f"相对误差：{relative_error:.2f}% (误差<10% 说明拟合有效)")

print("="*50)
print("耦合反馈方程系数校准完成")
print("="*50)
print(f"最终系数：α={ALPHA:.4f}, β={BETA:.4f}, ω={OMEGA:.4f}")

# ====================== 保存系数到文件 ======================
import json
coefficients = {
    'ALPHA': ALPHA,
    'BETA': BETA,
    'OMEGA': OMEGA
}
coeff_file = os.path.join(ROOT, 'sd', 'feedback', 'coupling_coefficients.json')
with open(coeff_file, 'w') as f:
    json.dump(coefficients, f, indent=4)
print(f"系数已保存到：{coeff_file}")