import pandas as pd
import numpy as np
import os

# Set up paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

csv_path = os.path.join(ROOT, 'sd', 'feedback', 'layers_0_23_batch_core_variables.csv')
df = pd.read_csv(csv_path)

df_agg = df.groupby('层').agg({
    '曲率 K(t)': 'mean',
    '曲率演化率 dK/dt(L)': lambda x: pd.to_numeric(x.replace('-', np.nan), errors='coerce').mean(),
    '语义漂移系数 S(t)': 'mean'
}).dropna()

print('数据层分布:')
print(f'总层数: {len(df_agg)}')
print(f'层范围: {df_agg.index.min()}-{df_agg.index.max()}')
print()

print('各层统计 (每5层显示一次):')
for layer in range(0, 24, 5):
    if layer in df_agg.index:
        row = df_agg.loc[layer]
        print('层{}: K={:.6f}, dK/dt={:.6f}, S={:.6f}'.format(
            layer,
            row['曲率 K(t)'],
            row['曲率演化率 dK/dt(L)'],
            row['语义漂移系数 S(t)']
        ))

print()
print('数据稳定性分析:')
print('dK/dt标准差: {:.8f}'.format(df_agg['曲率演化率 dK/dt(L)'].std()))
print('S标准差: {:.8f}'.format(df_agg['语义漂移系数 S(t)'].std()))

# 分析不同层段的稳定性
print()
print('不同层段稳定性分析:')
segments = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 23)]
for start, end in segments:
    segment_data = df_agg[(df_agg.index >= start) & (df_agg.index <= end)]
    if len(segment_data) > 0:
        dkdt_std = segment_data['曲率演化率 dK/dt(L)'].std()
        s_std = segment_data['语义漂移系数 S(t)'].std()
        print('层{}-{}: dK/dt_std={:.8f}, S_std={:.8f}, 样本数={}'.format(
            start, end, dkdt_std, s_std, len(segment_data)
        ))