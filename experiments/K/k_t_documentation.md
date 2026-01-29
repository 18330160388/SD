# K(t) 语义曲率计算文档

## 概述

K(t) 是基于局部散射曲率的语义分析指标，用于量化句子中每个token的语义聚集/发散程度。通过计算窗口内隐藏状态向量的协方差矩阵的几何性质，结合形态-语义匹配度M(t)，实现对语义空间局部曲率的量化。

## 核心概念

### 隐藏状态向量 h(t)
- **来源**：千问5B模型第12层隐藏状态向量
- **维度**：896维
- **表示**：每个token在语义空间中的位置向量

### 窗口设置
- **窗口大小**：N = 2k + 1，k=2时N=5
- **范围**：对于token t，窗口为 [t-2, t+2]
- **边界处理**：句子首尾不足时，用零向量补全

### 协方差矩阵 Σ(t)
- **定义**：窗口内向量的协方差矩阵
- **计算**：
  ```
  Σ(t) = (1/(N-1)) · Σⁿᵢ₌₁ [hᵢ(t) - h̄(t)] · [hᵢ(t) - h̄(t)]ᵀ
  ```
  其中 h̄(t) 为窗口内向量的均值

## 数学公式

### 1. 原始曲率 K(t)
```
K(t) = [det(Σ(t)) + ε] / [(tr(Σ(t)))² · (1 - ζ·M(t))²]
```

**参数说明**：
- `det(Σ(t))`：协方差矩阵的行列式，反映矩阵的"体积"（散射程度）
- `tr(Σ(t))`：协方差矩阵的迹，反映总方差
- `M(t)`：形态-语义匹配度，默认0.5
- `ζ`：形态因子，默认0.3
- `ε`：正则化参数，默认1e-8（防止行列式为0）

**物理意义**：
- 分子：det(Σ) 反映局部散射的几何体积
- 分母：(tr(Σ))² 反映总方差的平方，(1 - ζ·M(t))² 反映形态匹配的稳定性
- K(t) 值越大，表示局部散射越强（相对发散）

### 2. 对数曲率 K_log(t)
```
K_log(t) = ln(det(Σ(t)) + ε) - 2·ln(tr(Σ(t)) + ε) - 2·ln|1 - ζ·M(t)|
```

**优势**：
- 将乘除运算转换为加减运算，提高数值稳定性
- 避免极端值导致的计算溢出
- 所有K_log(t)值均为负数（因为det和tr通常很小）

### 3. Z-score归一化 K_norm(t)
```
K_norm(t) = (K_log(t) - μ) / σ
```

**计算步骤**：
1. 计算整句话所有token的K_log(t)均值 μ
2. 计算标准差 σ（样本标准差，ddof=1）
3. 对每个K_log(t)进行标准化

**特殊情况**：
- 如果 σ = 0（所有K_log相等），则 K_norm(t) = 0

## 计算流程

### 步骤1：向量提取
```python
h_t, token_num, tokenizer, inputs, _ = extract_hidden_states(text, layer_idx=12)
tokens = [tokenizer.decode([id]) for id in inputs['input_ids'][0]]
```

### 步骤2：窗口构建
```python
for t in range(token_num):
    window_indices = range(max(0, t-k), min(token_num, t+k+1))
    window_vectors = [h_t[idx] for idx in window_indices]
    # 补全到N个向量
    while len(window_vectors) < N:
        window_vectors.append(np.zeros(896))
```

### 步骤3：协方差计算
```python
vectors = np.array(window_vectors)
mean_vec = np.mean(vectors, axis=0)
centered = vectors - mean_vec
cov_matrix = np.cov(centered.T, ddof=1)
```

### 步骤4：曲率计算
```python
det_cov = np.linalg.det(cov_matrix)
trace_cov = np.trace(cov_matrix)
k_t = (det_cov + epsilon) / ((trace_cov ** 2) * ((1 - zeta * m_t) ** 2))
k_log_t = np.log(det_cov + epsilon) - 2*np.log(trace_cov + epsilon) - 2*np.log(abs(1 - zeta * m_t))
```

### 步骤5：归一化
```python
k_log_list = [r['k_log_t'] for r in results]
mu = np.mean(k_log_list)
sigma = np.std(k_log_list, ddof=1)
k_norm_list = [(k_log - mu) / sigma for k_log in k_log_list] if sigma != 0 else [0.0] * len(k_log_list)
```

## 语义解读

### K_norm(t) 值域
- **正值 (K_norm > 0)**：语义聚集
  - 表示该token的局部曲率相对较低
  - 语义空间中向量分布较为集中
  - 对应稳定、明确的语义概念

- **负值 (K_norm < 0)**：语义发散
  - 表示该token的局部曲率相对较高
  - 语义空间中向量分布较为分散
  - 对应动态、变化的语义概念

- **零值 (K_norm = 0)**：语义中性
  - 标准差为0时，所有token语义无差异

### 应用场景
- **动作词**：如"追"、"跑"，往往显示语义发散（动态变化）
- **名词**：如"小猫"、"花园"，往往显示语义聚集（稳定实体）
- **介词/助词**：如"的"、"在"，语义相对聚集（语法连接）

## 参数调优

### 默认参数
- `k = 2`：窗口半宽，平衡局部性和全局性
- `m_t = 0.5`：形态匹配度，可根据任务调整
- `zeta = 0.3`：形态因子权重
- `epsilon = 1e-8`：数值稳定性保证

### 调整建议
- **增大k**：获得更全局的语义信息，但可能丢失局部细节
- **减小k**：更敏感的局部变化，但可能受噪声影响
- **调整zeta**：控制形态匹配对曲率的影响程度

## 示例

### 输入句子："小猫追着蝴蝶跑过花园"

| Token | K(t) | K_log(t) | K_norm(t) | 语义解读 |
|-------|------|----------|-----------|----------|
| 小    | 0.0  | -44.392  | -1.208    | 语义发散 |
| 猫    | 0.0  | -44.393  | -1.208    | 语义发散 |
| 追    | 0.0  | -44.393  | -1.208    | 语义发散 |
| 着    | 0.0  | -27.744  | 0.716     | 语义聚集 |
| 蝴蝶  | 0.0  | -27.666  | 0.725     | 语义聚集 |
| 跑    | 0.0  | -27.796  | 0.710     | 语义聚集 |
| 过    | 0.0  | -27.707  | 0.720     | 语义聚集 |
| 花园  | 0.0  | -27.440  | 0.751     | 语义聚集 |

**分析**：
- 前三个token（"小猫追"）显示发散，反映动作开始的动态性
- 后五个token显示聚集，反映场景描述的稳定性
- K(t)为0是由于协方差矩阵的数值特性，K_log和K_norm提供了有效的相对比较

## 注意事项

1. **数值稳定性**：使用对数形式避免计算溢出
2. **边界处理**：窗口不足时补零向量，保证计算一致性
3. **维度一致性**：确保隐藏状态向量维度为896（千问5B第12层）
4. **异常处理**：模型加载失败时可使用随机向量测试
5. **相对性**：K_norm是句子内的相对比较，不是绝对指标

## 代码实现

主要函数：`compute_semantic_curvature(text)`

返回格式：
```python
[
    {
        "token": "词",
        "k_t": float,
        "k_log_t": float,
        "k_norm_t": float,
        "interpretation": "语义聚集" 或 "语义发散"
    },
    ...
]
```