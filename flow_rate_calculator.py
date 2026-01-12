"""
流率计算器 (Flow Rate Calculator)
计算系统动力学中的流量指标：dK/dt, dD/dt, dC/dt, dV/dt
理论依据：
1. 流率 = 状态变量对时间的导数 (Sterman, 2000)
2. 演化速率与上下文驱动力相关 (dV/dt ∝ -G(t))
"""

import torch
import numpy as np
from typing import List, Dict, Tuple


class FlowRateCalculator:
    """
    计算语义空间中各状态变量的演化速率（流率）
    
    核心方法：一阶差分近似导数
    dX/dt ≈ X(t+1) - X(t)
    """
    
    def __init__(self):
        """初始化流率计算器"""
        self.epsilon = 1e-8
    
    def compute_flow_rates(
        self,
        k_sequence: List[float],
        d_sequence: List[float],
        c_sequence: List[float],
        v_sequence: List[float],
        hidden_states: torch.Tensor = None
    ) -> Dict[str, np.ndarray]:
        """
        计算所有流率指标
        
        参数:
            k_sequence: K(t)曲率序列
            d_sequence: D(t)平均距离序列
            c_sequence: C(t)聚类密度序列
            v_sequence: V(t)语义势能序列
            hidden_states: 隐状态序列 [seq_len, hidden_dim]，用于计算驱动力G(t)
        
        返回:
            包含所有流率的字典:
            - 'dK_dt': 曲率演化速率
            - 'dD_dt': 距离变化速率
            - 'dC_dt': 聚类密度增长率
            - 'dV_dt': 势能衰减速率
            - 'G_t': 语义驱动力范数（如果提供hidden_states）
        """
        # 转换为numpy数组
        k_seq = np.array(k_sequence)
        d_seq = np.array(d_sequence)
        c_seq = np.array(c_sequence)
        v_seq = np.array(v_sequence)
        
        seq_len = len(k_seq)
        
        # 计算一阶差分（流率）
        # 注意：差分后长度为 seq_len - 1
        dK_dt = self._compute_first_difference(k_seq)
        dD_dt = self._compute_first_difference(d_seq)
        dC_dt = self._compute_first_difference(c_seq)
        dV_dt = self._compute_first_difference(v_seq)
        
        results = {
            'dK_dt': dK_dt,
            'dD_dt': dD_dt,
            'dC_dt': dC_dt,
            'dV_dt': dV_dt,
        }
        
        # 如果提供了隐状态，计算语义驱动力G(t)
        if hidden_states is not None:
            G_t = self._compute_semantic_driving_force(hidden_states)
            results['G_t'] = G_t
            
            # 计算dV/dt与-G(t)的相关性（验证理论）
            if len(dV_dt) == len(G_t):
                correlation = self._compute_correlation(dV_dt, -G_t)
                results['dV_G_correlation'] = correlation
        
        return results
    
    def _compute_first_difference(self, sequence: np.ndarray) -> np.ndarray:
        """
        计算一阶差分（离散导数）
        dX/dt ≈ X(t+1) - X(t)
        
        参数:
            sequence: 长度为n的序列
        
        返回:
            长度为n-1的差分序列
        """
        if len(sequence) < 2:
            return np.array([])
        
        # 计算相邻元素的差值
        diff = np.diff(sequence)
        return diff
    
    def _compute_semantic_driving_force(self, hidden_states: torch.Tensor) -> np.ndarray:
        """
        计算语义驱动力 G(t) = ||h(t+1) - h(t)||₂
        这是状态更新的幅度，理论上与dV/dt负相关
        
        参数:
            hidden_states: [seq_len, hidden_dim]
        
        返回:
            长度为seq_len-1的驱动力序列
        """
        if hidden_states is None or hidden_states.size(0) < 2:
            return np.array([])
        
        # 确保是tensor
        if isinstance(hidden_states, np.ndarray):
            hidden_states = torch.from_numpy(hidden_states).float()
        
        seq_len = hidden_states.size(0)
        G_t = []
        
        for t in range(seq_len - 1):
            # 计算状态变化向量
            delta_h = hidden_states[t+1] - hidden_states[t]
            
            # 计算L2范数（欧氏距离）
            driving_force = torch.norm(delta_h, p=2).item()
            G_t.append(driving_force)
        
        return np.array(G_t)
    
    def _compute_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        计算两个序列的Pearson相关系数
        用于验证 dV/dt ∝ -G(t) 的理论关系
        
        参数:
            x: 序列1 (通常是dV/dt)
            y: 序列2 (通常是-G(t))
        
        返回:
            相关系数 [-1, 1]
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        # 计算Pearson相关系数
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
        
        if denominator < self.epsilon:
            return 0.0
        
        correlation = numerator / denominator
        return correlation
    
    def compute_flow_statistics(self, flow_rates: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        计算流率的统计信息
        
        参数:
            flow_rates: 流率字典
        
        返回:
            每个流率的统计信息（均值、最大、最小、标准差）
        """
        statistics = {}
        
        for name, values in flow_rates.items():
            if name == 'dV_G_correlation':
                continue  # 跳过相关系数
            
            if len(values) == 0:
                statistics[name] = {
                    'mean': 0.0,
                    'max': 0.0,
                    'min': 0.0,
                    'std': 0.0
                }
            else:
                statistics[name] = {
                    'mean': float(np.mean(values)),
                    'max': float(np.max(values)),
                    'min': float(np.min(values)),
                    'std': float(np.std(values))
                }
        
        return statistics
    
    def identify_critical_points(
        self,
        flow_rates: Dict[str, np.ndarray],
        threshold_percentile: float = 90.0
    ) -> Dict[str, List[int]]:
        """
        识别流率的临界点（极值点）
        这些点通常对应语义的"质变时刻"
        
        参数:
            flow_rates: 流率字典
            threshold_percentile: 阈值百分位数（默认90%，即top 10%）
        
        返回:
            每个流率的临界点位置列表
        """
        critical_points = {}
        
        for name, values in flow_rates.items():
            if name == 'dV_G_correlation' or len(values) == 0:
                continue
            
            # 计算绝对值（关注变化幅度）
            abs_values = np.abs(values)
            
            # 计算阈值
            threshold = np.percentile(abs_values, threshold_percentile)
            
            # 找到超过阈值的位置
            critical_indices = np.where(abs_values >= threshold)[0].tolist()
            critical_points[name] = critical_indices
        
        return critical_points
    
    def analyze_flow_patterns(
        self,
        flow_rates: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        分析流率的演化模式
        
        参数:
            flow_rates: 流率字典
        
        返回:
            模式分析结果：
            - positive_ratio: 正值占比（增长趋势）
            - negative_ratio: 负值占比（下降趋势）
            - volatility: 波动性（标准差/均值绝对值）
        """
        patterns = {}
        
        for name, values in flow_rates.items():
            if name == 'dV_G_correlation' or len(values) == 0:
                continue
            
            # 正负比例
            positive_count = np.sum(values > 0)
            negative_count = np.sum(values < 0)
            total_count = len(values)
            
            positive_ratio = positive_count / total_count if total_count > 0 else 0.0
            negative_ratio = negative_count / total_count if total_count > 0 else 0.0
            
            # 波动性
            mean_abs = np.mean(np.abs(values))
            std = np.std(values)
            volatility = std / (mean_abs + self.epsilon)
            
            patterns[name] = {
                'positive_ratio': float(positive_ratio),
                'negative_ratio': float(negative_ratio),
                'volatility': float(volatility)
            }
        
        return patterns


def init_flow_calculator() -> FlowRateCalculator:
    """
    初始化流率计算器
    
    返回:
        FlowRateCalculator实例
    """
    return FlowRateCalculator()


# 测试代码
if __name__ == "__main__":
    # 模拟数据
    k_seq = [0.2, 0.3, 0.8, 0.7, 0.6, 0.5]
    d_seq = [0.5, 0.6, 0.9, 0.7, 0.6, 0.5]
    c_seq = [0.8, 0.7, 0.3, 0.5, 0.6, 0.7]
    v_seq = [0.9, 0.8, 0.5, 0.3, 0.2, 0.2]
    
    # 模拟隐状态
    hidden_states = torch.randn(6, 896)
    
    # 初始化计算器
    calculator = init_flow_calculator()
    
    # 计算流率
    flow_rates = calculator.compute_flow_rates(k_seq, d_seq, c_seq, v_seq, hidden_states)
    
    # 打印结果
    print("流率计算结果：")
    for name, values in flow_rates.items():
        if name != 'dV_G_correlation':
            print(f"{name}: {values}")
        else:
            print(f"{name}: {values:.4f}")
    
    # 统计信息
    stats = calculator.compute_flow_statistics(flow_rates)
    print("\n统计信息：")
    for name, stat in stats.items():
        print(f"{name}: {stat}")
    
    # 临界点
    critical = calculator.identify_critical_points(flow_rates)
    print("\n临界点（top 10%）：")
    for name, indices in critical.items():
        print(f"{name}: {indices}")
    
    # 模式分析
    patterns = calculator.analyze_flow_patterns(flow_rates)
    print("\n演化模式：")
    for name, pattern in patterns.items():
        print(f"{name}: {pattern}")
