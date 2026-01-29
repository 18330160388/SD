# S(t) 语义漂移系数计算公式

## 一、公式定义

$$
S(t) = \frac{\text{dist}(\mathbf{h}(t), \mathbf{h}_{\text{global}})}{\text{dist}_{\max}(t)} \times [1 - \omega \cdot C(t) + \nu \cdot D(t)] \times [1 - \mu \cdot M(t)]
$$

**语义**：
- **S(t) → 0**：与全局主题一致，无漂移
- **S(t) → 1**：偏离全局主题，漂移显著

**参数**：$\omega=0.3$，$\nu=0.2$，$\mu=0.25$，$\lambda=0.1$

---

## 二、计算流程

### 1. 提取语义向量 h(t)

从LLM中间层（Layer 12）提取：
```python
hidden_states, _, tokenizer, inputs, _ = extract_hidden_states(
    text=text,
    model_name="Qwen2.5-0.5B-Instruct",
    middle_layer_idx=12
)
```
- 维度：$\mathbf{h}(t) \in \mathbb{R}^{896}$
- 包含语义与上下文信息

### 2. **【改进1】L2归一化**

$$
\mathbf{h}_{\text{norm}}(t) = \frac{\mathbf{h}(t)}{\|\mathbf{h}(t)\|_2}
$$

```python
if self.normalize_hidden:
    hidden_states = F.normalize(hidden_states, p=2, dim=1)
```

**物理意义**：
- **问题**：首token范数异常（江=1579 vs 河=16.74）
- **解决**：归一化到单位球面（所有范数=1）
- **效果**：距离反映**方向差异**而非数值大小

### 3. **【改进2】全局锚点**

#### 原始动态锚点（use_global_anchor=False）

$$
\mathbf{h}_{\text{global}}(t) = \sum_{s=0}^{t} \alpha(s,t) \cdot \mathbf{h}(s), \quad \alpha(s,t) = \text{softmax}\left( \frac{\mathbf{h}(s)^T \cdot \mathbf{h}(t)}{\sqrt{d} \cdot \exp(\lambda \cdot (t-s))} \right)
$$

- 因果，只看历史$s \leq t$
- 每个$t$的锚点不同
- 易被首token污染

#### 全局锚点（use_global_anchor=True，推荐）

$$
\mathbf{h}_{\text{global}} = \sum_{i=0}^{T-1} w(i) \cdot \mathbf{h}(i), \quad w(i) = \text{softmax}\left( \frac{1}{T}\sum_{j=0}^{T-1} \mathbf{h}(i)^T \cdot \mathbf{h}(j) \right)
$$

```python
def _compute_sentence_embedding(h_sequence):
    # 计算每个token与所有token的平均相似度
    similarity_matrix = h_sequence @ h_sequence.T  # [T, T]
    avg_similarity = similarity_matrix.mean(dim=1)  # [T]
    weights = F.softmax(avg_similarity, dim=0)
    h_global = (h_sequence * weights.unsqueeze(1)).sum(dim=0)
    return h_global
```

**物理意义**：
- **改进**：所有token共享同一个锚点（代表整句主题）
- **权重**：与全句相似度高的词（如核心主题词）权重大
- **稳定**：标准差0.111 vs 原始0.461

### 4. 计算距离比例

$$
\text{dist\_ratio}(t) = \frac{\|\mathbf{h}(t) - \mathbf{h}_{\text{global}}\|_2}{\text{dist}_{\max}(t)}
$$

其中 $\text{dist}_{\max}(t) = \max_{s \in [0,t]} \|\mathbf{h}(s) - \mathbf{h}_{\text{global}}\|_2$

```python
dist = torch.norm(h_current - h_global, p=2)
dist_max = max([torch.norm(h_sequence[s] - h_global, p=2) for s in range(t+1)])
dist_ratio = dist / (dist_max + 1e-8)
```

### 5. 计算M(t)、C(t)、D(t)

#### M(t)：形态-语义匹配度
```python
m_t = compute_m_t_full(
    h_t=hidden_states[t],
    token_text=tokens[t],
    tokens=tokens,
    token_idx=t,
    hidden_states=hidden_states,
    layer_idx=12
)
```
- 调用 `m_t_calculator.compute_m_t_full()`
- 公式：$M(t) = \cos(\mathbf{h}(t), \Phi(\mathbf{m}(t)))$

#### C(t)：聚类密度
```python
c_t = compute_c_t(
    h_t=hidden_states[t],
    hidden_states=hidden_states,
    token_idx=t,
    k=3,
    theta=0.5,
    alpha=0.4,
    precomputed_m_t=m_t
)
```
- 调用 `c_t_calculator.compute_c_t()`
- 已有独立公式定义

#### D(t)：平均欧氏距离
```python
d_t = compute_d_t(
    h_t=hidden_states[t],
    hidden_states=hidden_states,
    token_idx=t,
    sentence_length=len(tokens),
    window_size=3,
    sim_threshold=0.5,
    precomputed_m_t=m_t
)
```
- 调用 `d_t_calculator.compute_d_t()`
- 已有独立公式定义

### 6. 计算S(t)

$$
S(t) = \text{dist\_ratio}(t) \times [1 - \omega \cdot C(t) + \nu \cdot D(t)] \times [1 - \mu \cdot M(t)]
$$

```python
stability_term = 1.0 - self.omega * c_t + self.nu * d_t
xi = 1.0 - self.mu * m_t
s_t_value = dist_ratio * stability_term * xi
S_t[t] = torch.clamp(s_t_value, 0.0, 1.0)
```

---

## 三、关键改进及物理意义

### 改进1：L2归一化 (normalize_hidden=True)

**问题**：首token（如"江"）范数异常（1579 vs 河=16.74）

**方案**：$\mathbf{h}_{\text{norm}} = \mathbf{h} / \|\mathbf{h}\|_2$

**物理意义**：
- 将所有向量投影到单位超球面（范数=1）
- 距离计算反映**方向差异**而非**数值大小**
- 消除模型中间层首token的表示偏差

### 改进2：全局锚点 (use_global_anchor=True)

**问题**：动态锚点$h_{global}(t)$随t变化，易被首token污染

**方案**：固定锚点 $h_{global} = \sum w(i) \cdot h(i)$，权重基于平均相似度

**物理意义**：
- 所有token与**同一个主题基准**比较
- 权重自动聚焦到核心主题词（与全句相似度高）
- 真正度量"偏离全局主题"而非"偏离历史趋势"
- 结果更稳定（std: 0.111 vs 0.461）

---

## 四、实现代码

```python
class SemanticDriftCoeff(nn.Module):
    def __init__(self, lambda_decay=0.1, normalize_hidden=True, 
                 use_global_anchor=False):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(0.25))
        self.omega = 0.3
        self.nu = 0.2
        self.lambda_decay = lambda_decay
        self.normalize_hidden = normalize_hidden
        self.use_global_anchor = use_global_anchor
        
    def forward(self, text):
        # 1. 提取hidden states
        hidden_states = extract_hidden_states(text, layer=12)
        
        # 2. L2归一化
        if self.normalize_hidden:
            hidden_states = F.normalize(hidden_states, p=2, dim=1)
        
        # 3. 计算全局锚点
        if self.use_global_anchor:
            h_global_fixed = self._compute_sentence_embedding(hidden_states)
        
        # 4. 对每个token计算S(t)
        S_t = torch.zeros(len(hidden_states))
        for t in range(len(hidden_states)):
            h_current = hidden_states[t]
            
            # 锚点
            if self.use_global_anchor:
                h_global = h_global_fixed
            else:
                h_global = self._compute_global_anchor(hidden_states, t)
            
            # 距离
            dist = torch.norm(h_current - h_global, p=2)
            dist_max = self._compute_dist_max(hidden_states, t)
            dist_ratio = dist / (dist_max + 1e-8)
            
            # M(t), C(t), D(t)
            m_t = self._compute_m_t(hidden_states, tokens, t, tokenizer)
            c_t = self._compute_c_t(hidden_states, tokens, t, m_t)
            d_t = self._compute_d_t(hidden_states, tokens, t, m_t)
            
            # S(t)
            stability_term = 1.0 - self.omega * c_t + self.nu * d_t
            xi = 1.0 - self.mu * m_t
            s_t_value = dist_ratio * stability_term * xi
            S_t[t] = torch.clamp(s_t_value, 0.0, 1.0)
        
        return S_t
```

---

## 五、参数配置

| 参数 | 取值 | 说明 |
|-----|-----|-----|
| λ (lambda_decay) | 0.1 | 动态锚点的时间衰减 |
| μ (mu) | 0.25 | 形态权重（消融实验优化）|
| ω (omega) | 0.3 | 聚类密度权重 |
| ν (nu) | 0.2 | 平均距离权重 |
| d_model | 896 | Qwen2.5-0.5B维度 |
| layer | 12 | 中间层索引 |

---

## 六、模式对比

| 维度 | 动态锚点 | 全局锚点（推荐）|
|-----|---------|--------------|
| 锚点 | $h_{global}(t)$，每个t不同 | $h_{global}$，固定 |
| 因果性 | 因果（s≤t）| 非因果（全句）|
| 稳定性 | std=0.461 | **std=0.111** |
| 语义 | 偏离历史趋势 | **偏离全局主题** |

---

## 七、测试结果："江河湖海都是水"

| Token | S(t) | 解释 |
|-------|------|-----|
| 江 | 0.515 | 首token，接近主题 |
| 河 | 0.840 | 最大漂移 |
| 湖 | 0.615 | 中等漂移 |
| 海 | 0.599 | 中等漂移 |
| 都是 | 0.722 | 连词，较高漂移 |
| 水 | 0.544 | 主题词，语义稳定 |

**统计**：mean=0.639, std=0.111, min=0.515, max=0.840
