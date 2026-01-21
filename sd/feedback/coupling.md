# 语义演化规律：核心回路公式与逻辑文档
## 一、最终核心公式（纯机制驱动，无模糊项）
$$
\boxed{\frac{dK}{dt}(L) = \alpha \cdot \frac{dK}{dt}(L-1) - \beta \cdot \left( S(L-1)^2 + \omega \cdot |\Delta D(L-1)| \right)}
$$

### 符号定义（简洁版）
| 符号                | 含义                          | 物理意义                          |
|---------------------|-------------------------------|-----------------------------------|
| $\frac{dK}{dt}(L)$  | 当前层（L）曲率演化速率        | 语义聚合变化的快慢                |
| $\frac{dK}{dt}(L-1)$| 上一层（L-1）曲率演化速率      | 前序层语义聚合的惯性驱动源        |
| $S(L-1)$            | 上一层（L-1）语义漂移系数      | 前序层语义偏离核心的程度          |
| $|\Delta D(L-1)|$   | 上一层（L-1）平均欧氏距离变化绝对值 | 前序层语义分布的波动幅度          |
| $\alpha/\beta/\omega$ | 拟合系数（均为正数）          | 分别代表驱动强度、总抑制强度、距离抑制权重 |

## 二、核心耦合反馈回路（因果逻辑闭环）
$$
\underbrace{\frac{dK}{dt}(L-1)}_{上一层驱动} \xrightarrow{+} \underbrace{\frac{dK}{dt}(L)}_{当前层速率} \xrightarrow{+} \begin{cases} 
\underbrace{S(L)}_{当前层语义漂移} \xrightarrow{+} \underbrace{S(L)^2}_{非线性抑制} \xrightarrow{+} \underbrace{\text{总抑制项}}_{反馈抑制} \xrightarrow{-} \underbrace{\frac{dK}{dt}(L+1)}_{下一层速率} \\
\underbrace{|\Delta D(L)|}_{当前层距离变化} \xrightarrow{+} \underbrace{\text{总抑制项}}_{反馈抑制} \xrightarrow{-} \underbrace{\frac{dK}{dt}(L+1)}_{下一层速率}
\end{cases}
$$

### 回路核心逻辑
1. **驱动机制**：上一层语义聚合的节奏（$\frac{dK}{dt}(L-1)$）正向传递，决定当前层的基础聚合速率；
2. **抑制机制**：上一层的语义漂移（$S(L-1)^2$）和距离波动（$|\Delta D(L-1)|$）共同构成负反馈，抑制当前层速率过快，避免语义发散；
3. **演化本质**：语义演化是“层间驱动”与“双重抑制”的动态平衡，实现聚合与稳定的统一。

要不要我帮你整理一份**公式拟合操作指引**，包含系数约束条件、数据输入格式和结果解读模板，方便直接落地计算？