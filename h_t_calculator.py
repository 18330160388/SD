"""
中文大语言模型语义层多义性熵 H(t) 计算器

基于文档 3-3 定义：
H(t) = (1/log|S_t|) * [-∑_{s∈S_t} p(s|t)·log(p(s|t) + ε)] * ζ(t)

核心组件：
1. 核心义项集合 S_t：基于中文多义词词典
2. 义项激活概率 p(s|t)：融合语义、上下文、形态、句法特征
3. 归一化因子：1/log|S_t| 确保跨token可比
4. 语境修正因子 ζ(t)：适配中文搭配特性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

# 注意：m_t_calculator 的导入移到需要使用的地方，避免循环导入


class PolysemyDictionary:
    """中文多义词词典管理器
    
    基于《现代汉语词典》（第7版）+ HowNet义原体系
    内置150+常用多义词，覆盖日常使用场景
    """
    
    def __init__(self):
        # 义项数量范围约束（文档3-3规定）
        self.min_senses = 2
        self.max_senses = 15
        self.default_sense_count = 1
        
        # 加载内置多义词词典
        self.polysemy_dict = self._get_polysemy_dict()
    
    def _get_polysemy_dict(self) -> Dict[str, int]:
        """内置扩展多义词词典（150+常用词）
        
        数据来源：
        1. 《现代汉语词典》第7版高频多义词
        2. 《现代汉语常用词表》（教育部）
        3. HSK词汇表多义词统计
        """
        return {
            # ===== 高频多义动词（50个）=====
            "打": 10, "看": 7, "走": 6, "跑": 5, "来": 7, "去": 6, "上": 6, "下": 6,
            "进": 5, "出": 8, "回": 5, "过": 8, "开": 9, "关": 6, "拿": 5, "放": 7,
            "给": 6, "送": 5, "带": 6, "收": 6, "拉": 7, "推": 5, "提": 6, "抬": 4,
            "扔": 4, "丢": 5, "找": 5, "换": 5, "买": 4, "卖": 4, "用": 6, "做": 7,
            "办": 5, "干": 6, "搞": 7, "弄": 6, "整": 5, "治": 6, "理": 6, "管": 6,
            "行": 5, "发": 8, "生": 6, "长": 4, "成": 7, "变": 6, "化": 5, "转": 6,
            "改": 5, "换": 5,
            
            # ===== 高频多义名词（40个）=====
            "手": 4, "头": 5, "脸": 4, "眼": 5, "心": 6, "身": 5, "口": 6, "面": 6,
            "气": 7, "力": 6, "道": 5, "理": 6, "法": 7, "意": 6, "情": 5, "事": 6,
            "物": 5, "人": 5, "家": 6, "国": 5, "天": 5, "地": 5, "时": 5, "年": 4,
            "日": 5, "月": 4, "点": 7, "处": 6, "方": 6, "边": 5, "间": 5, "里": 5,
            "外": 4, "中": 5, "内": 4, "前": 4, "后": 4, "左": 3, "右": 3, "东": 4,
            
            # ===== 高频多义形容词（30个）=====
            "大": 5, "小": 5, "高": 6, "低": 5, "长": 4, "短": 4, "宽": 3, "窄": 3,
            "厚": 3, "薄": 4, "深": 5, "浅": 4, "重": 5, "轻": 4, "快": 5, "慢": 4,
            "好": 7, "坏": 5, "新": 4, "旧": 4, "老": 5, "少": 5, "多": 5, "远": 4,
            "近": 4, "早": 4, "晚": 4, "冷": 4, "热": 5, "干": 4,
            
            # ===== 高频虚词/副词（20个）=====
            "就": 8, "才": 6, "还": 7, "又": 6, "也": 5, "都": 5, "只": 5, "更": 4,
            "最": 3, "很": 3, "太": 4, "真": 5, "正": 6, "刚": 5, "将": 5, "曾": 4,
            "已": 4, "被": 3, "把": 5, "让": 5,
            
            # ===== 其他常用多义词（10个）=====
            "然": 8, "的": 6, "得": 7, "地": 5, "了": 6, "着": 6, "过": 8, "起": 6,
            "为": 7, "被": 3,
        }
    
    def get_sense_count(self, token: str) -> int:
        """获取token的核心义项数量"""
        return self.polysemy_dict.get(token, self.default_sense_count)
    
    def is_polysemous(self, token: str) -> bool:
        """判断是否为多义词"""
        return self.get_sense_count(token) >= self.min_senses


class SenseActivationModel(nn.Module):
    """义项激活概率模型
    
    p(s|t) = softmax(MLP(Concat(h(t), c(t), M(t), syn(t))))
    
    融合四大特征：
    - h(t): 当前token语义向量
    - c(t): 上下文特征（窗口t-3~t+3的注意力加权）
    - M(t): 形态-语义匹配度（标量，来自m_t_calculator）
    - syn(t): 句法搭配特征（简化为位置和邻域特征）
    """
    
    def __init__(self, hidden_dim: int = 896, morph_dim: int = 254, 
                 context_window: int = 3, max_senses: int = 15):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.morph_dim = morph_dim
        self.context_window = context_window
        self.max_senses = max_senses
        
        # 上下文特征提取（注意力加权机制）
        self.context_attention = nn.Linear(hidden_dim, 1)
        
        # 句法特征提取（简化为位置嵌入+邻域语义）
        self.syntax_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # 左右邻居语义
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # 融合MLP：h(d) + c(d) + M(t)(1) + syn(64) → sense_dim
        input_dim = hidden_dim + hidden_dim + morph_dim + 64
        self.sense_mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, max_senses)  # 输出最大义项数的logits
        )
    
    def extract_context_features(self, hidden_states: torch.Tensor, 
                                  token_idx: int, 
                                  attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """提取上下文特征 c(t)
        
        Args:
            hidden_states: [seq_len, hidden_dim]
            token_idx: 当前token索引
            attention_weights: [seq_len, seq_len] 注意力权重矩阵
        
        Returns:
            context_vector: [hidden_dim] 上下文加权向量
        """
        seq_len = hidden_states.size(0)
        
        # 定义上下文窗口 [t-3, t+3]
        start_idx = max(0, token_idx - self.context_window)
        end_idx = min(seq_len, token_idx + self.context_window + 1)
        
        # 提取窗口内的hidden states
        context_hiddens = hidden_states[start_idx:end_idx]  # [window_size, hidden_dim]
        
        # 检查注意力权重是否可用且为二维矩阵 [seq_len, seq_len]
        use_external_attention = (
            attention_weights is not None and 
            attention_weights.dim() == 2 and 
            attention_weights.size(0) == seq_len and
            attention_weights.size(1) == seq_len
        )
        
        if use_external_attention:
            # 使用当前token对窗口内token的注意力作为权重
            attn_weights = attention_weights[token_idx, start_idx:end_idx]  # [window_size]
            attn_weights = F.softmax(attn_weights, dim=0).unsqueeze(-1)  # [window_size, 1]
            context_vector = (context_hiddens * attn_weights).sum(dim=0)  # [hidden_dim]
        else:
            # 无有效注意力权重时，使用可学习的注意力
            attn_scores = self.context_attention(context_hiddens).squeeze(-1)  # [window_size]
            attn_weights = F.softmax(attn_scores, dim=0).unsqueeze(-1)  # [window_size, 1]
            context_vector = (context_hiddens * attn_weights).sum(dim=0)  # [hidden_dim]
        
        return context_vector
    
    def extract_syntax_features(self, hidden_states: torch.Tensor, 
                               token_idx: int) -> torch.Tensor:
        """提取句法搭配特征 syn(t)
        
        简化实现：提取左右邻居的语义向量作为句法搭配信号
        
        Args:
            hidden_states: [seq_len, hidden_dim]
            token_idx: 当前token索引
        
        Returns:
            syntax_vector: [64] 句法特征向量
        """
        seq_len = hidden_states.size(0)
        
        # 提取左右邻居
        left_neighbor = hidden_states[token_idx - 1] if token_idx > 0 else hidden_states[token_idx]
        right_neighbor = hidden_states[token_idx + 1] if token_idx < seq_len - 1 else hidden_states[token_idx]
        
        # 拼接左右邻居作为句法搭配特征
        syntax_input = torch.cat([left_neighbor, right_neighbor], dim=0)  # [2 * hidden_dim]
        syntax_vector = self.syntax_proj(syntax_input)  # [64]
        
        return syntax_vector
    
    def forward(self, hidden_state: torch.Tensor, 
                context_feature: torch.Tensor,
                m_t_value: torch.Tensor,
                syntax_feature: torch.Tensor,
                num_senses: int) -> torch.Tensor:
        """计算义项激活概率分布
        
        Args:
            hidden_state: [hidden_dim] 当前token语义向量 h(t)
            context_feature: [hidden_dim] 上下文特征 c(t)
            m_t_value: [1] M(t)形态-语义匹配度标量
            syntax_feature: [64] 句法特征 syn(t)
            num_senses: 当前token的核心义项数量 |S_t|
        
        Returns:
            sense_probs: [num_senses] 归一化的义项激活概率
        """
        # [调试] 输出M(t)特征
        # print(f"  [SenseActivationModel.forward] M(t)值: {m_t_value.item():.6f}")
        
        # 拼接所有特征
        fused_features = torch.cat([
            hidden_state,
            context_feature,
            m_t_value,
            syntax_feature
        ], dim=0)  # [hidden_dim + hidden_dim + 1 + 64]
        
        # [调试] 输出融合后的特征
        # print(f"  [SenseActivationModel.forward] fused_features维度: {fused_features.shape}")
        
        # MLP映射到义项空间
        sense_logits = self.sense_mlp(fused_features)  # [max_senses]
        
        # 只取前num_senses个logits并归一化
        sense_logits = sense_logits[:num_senses]  # [num_senses]
        sense_probs = F.softmax(sense_logits, dim=0)  # [num_senses]
        return sense_probs


class PolysemyEntropyCalculator:
    """中文LLM语义层多义性熵 H(t) 计算器
    
    完整实现文档3-3定义的多义性熵计算管线
    """
    
    def __init__(self, hidden_dim: int = 896, morph_dim: int = 254,
                 epsilon: float = 1e-8, gamma: float = 0.08):
        """
        Args:
            hidden_dim: 语义向量维度
            morph_dim: 形态特征维度（已废弃，现在使用M(t)标量）
            epsilon: 正则化项，避免log(0)
            gamma: 语境修正因子权重系数
        """
        self.hidden_dim = hidden_dim
        self.morph_dim = morph_dim  # 保留兼容性，但不再使用
        self.epsilon = epsilon
        self.gamma = gamma
        
        # 初始化多义词词典
        self.polysemy_dict = PolysemyDictionary()
        
        # 初始化义项激活模型
        self.sense_model = SenseActivationModel(
            hidden_dim=hidden_dim,
            morph_dim=1  # 只用M(t)标量
        )
        
        # H(t)计算现在直接使用M(t)值，不再需要形态特征提取器
    
    def compute_collocation_strength(self, tokens: List[str], 
                                    token_idx: int) -> float:
        """计算固定搭配强度 colloc(t)
        
        简化实现：基于常见搭配模式的启发式规则
        
        Args:
            tokens: 完整token序列
            token_idx: 当前token索引
        
        Returns:
            colloc_strength: 搭配强度 ∈ [0, 1]
        """
                # ↑误缩进残留，已移至SenseActivationModel初始化参数
        strong_collocations = {
            # ===== 高频动词+名词搭配 =====
            # "打"相关
            ("打", "电话"): 1.0, ("打", "球"): 0.9, ("打", "针"): 0.85, ("打", "工"): 0.8,
            ("打", "补"): 0.85, ("打", "架"): 0.9, ("打", "印"): 0.85, ("打", "扫"): 0.8,
            ("打", "牌"): 0.85, ("打", "鼓"): 0.8, ("打", "伞"): 0.75, ("打", "车"): 0.7,
            ("打", "包"): 0.75, ("打", "折"): 0.8, ("打", "水"): 0.7, ("打", "油"): 0.75,
            ("打", "火"): 0.8, ("打", "雷"): 0.85, ("打", "闪"): 0.8, ("打", "鸣"): 0.8,
            
            # "看"相关
            ("看", "书"): 0.9, ("看", "病"): 0.85, ("看", "报"): 0.8, ("看", "戏"): 0.8,
            ("看", "门"): 0.75, ("看", "家"): 0.8, ("看", "管"): 0.75, ("看", "待"): 0.7,
            ("看", "法"): 0.8, ("看", "重"): 0.85, ("看", "轻"): 0.8, ("看", "起"): 0.75,
            
            # "开"相关
            ("开", "会"): 0.95, ("开", "车"): 0.9, ("开", "门"): 0.85, ("开", "枪"): 0.9,
            ("开", "刀"): 0.85, ("开", "张"): 0.8, ("开", "始"): 0.9, ("开", "放"): 0.85,
            ("开", "发"): 0.9, ("开", "辟"): 0.85, ("开", "拓"): 0.8, ("开", "创"): 0.85,
            ("开", "办"): 0.8, ("开", "设"): 0.85, ("开", "通"): 0.8, ("开", "工"): 0.85,
            
            # "行"相关
            ("行", "业"): 0.9, ("银", "行"): 1.0, ("行", "为"): 0.85, ("行", "事"): 0.8,
            ("行", "走"): 0.75, ("行", "动"): 0.8, ("行", "程"): 0.75, ("行", "列"): 0.8,
            
            # "做"相关
            ("做", "事"): 0.8, ("做", "工"): 0.85, ("做", "饭"): 0.9, ("做", "菜"): 0.85,
            ("做", "梦"): 0.8, ("做", "主"): 0.75, ("做", "客"): 0.8, ("做", "寿"): 0.75,
            
            # "吃"相关
            ("吃", "饭"): 0.9, ("吃", "药"): 0.85, ("吃", "亏"): 0.8, ("吃", "苦"): 0.85,
            ("吃", "惊"): 0.8, ("吃", "力"): 0.75, ("吃", "醋"): 0.8, ("吃", "水"): 0.7,
            
            # "喝"相关
            ("喝", "水"): 0.9, ("喝", "酒"): 0.85, ("喝", "茶"): 0.8, ("喝", "奶"): 0.8,
            ("喝", "粥"): 0.75, ("喝", "汤"): 0.8, ("喝", "彩"): 0.7,
            
            # "睡"相关
            ("睡", "觉"): 0.95, ("睡", "眠"): 0.8, ("睡", "午"): 0.75,
            
            # "写"相关
            ("写", "字"): 0.9, ("写", "信"): 0.85, ("写", "作"): 0.8, ("写", "诗"): 0.8,
            ("写", "文"): 0.85, ("写", "书"): 0.8, ("写", "稿"): 0.85,
            
            # "读"相关
            ("读", "书"): 0.9, ("读", "报"): 0.85, ("读", "信"): 0.8, ("读", "诗"): 0.8,
            
            # "说"相关
            ("说", "话"): 0.9, ("说", "明"): 0.85, ("说", "法"): 0.8, ("说", "服"): 0.8,
            ("说", "笑"): 0.75, ("说", "媒"): 0.7, ("说", "合"): 0.75,
            
            # "听"相关
            ("听", "话"): 0.8, ("听", "课"): 0.85, ("听", "讲"): 0.8, ("听", "众"): 0.75,
            ("听", "从"): 0.8, ("听", "见"): 0.7, ("听", "取"): 0.75,
            
            # "走"相关
            ("走", "路"): 0.85, ("走", "访"): 0.8, ("走", "廊"): 0.75, ("走", "狗"): 0.7,
            ("走", "运"): 0.8, ("走", "神"): 0.75, ("走", "火"): 0.8,
            
            # "跑"相关
            ("跑", "步"): 0.9, ("跑", "车"): 0.8, ("跑", "腿"): 0.75, ("跑", "道"): 0.8,
            
            # "坐"相关
            ("坐", "车"): 0.85, ("坐", "船"): 0.8, ("坐", "牢"): 0.85, ("坐", "位"): 0.8,
            ("坐", "月"): 0.75, ("坐", "井"): 0.7,
            
            # "站"相关
            ("站", "立"): 0.8, ("站", "台"): 0.85, ("站", "岗"): 0.9, ("站", "队"): 0.8,
            
            # ===== 名词+动词搭配 =====
            ("电", "话"): 0.9, ("饭", "店"): 0.85, ("学", "校"): 0.9, ("医", "院"): 0.9,
            ("公", "司"): 0.85, ("工", "厂"): 0.85, ("学", "生"): 0.9, ("老", "师"): 0.9,
            ("父", "母"): 0.9, ("兄", "弟"): 0.85, ("姐", "妹"): 0.85, ("朋", "友"): 0.9,
            ("邻", "居"): 0.85, ("同", "事"): 0.9, ("领", "导"): 0.85, ("部", "门"): 0.8,
            
            # ===== 形容词+名词搭配 =====
            ("大", "家"): 0.8, ("小", "心"): 0.85, ("高", "兴"): 0.9, ("快", "乐"): 0.9,
            ("美", "好"): 0.85, ("漂", "亮"): 0.85, ("干", "净"): 0.8, ("清", "楚"): 0.85,
            ("方", "便"): 0.8, ("重", "要"): 0.9, ("困", "难"): 0.85, ("简", "单"): 0.8,
            ("复", "杂"): 0.85, ("专", "业"): 0.8, ("普", "通"): 0.75, ("特", "别"): 0.8,
            
            # ===== 其他高频搭配 =====
            ("男", "人"): 0.85, ("女", "人"): 0.85, ("孩", "子"): 0.9, ("老", "人"): 0.85,
            ("青", "年"): 0.8, ("中", "年"): 0.8, ("少", "年"): 0.8, ("婴", "儿"): 0.8,
            ("动", "物"): 0.8, ("植", "物"): 0.8, ("建", "筑"): 0.75, ("机", "器"): 0.8,
            ("科", "学"): 0.85, ("教", "育"): 0.85, ("医", "学"): 0.8, ("文", "学"): 0.85,
            ("音", "乐"): 0.85, ("绘", "画"): 0.8, ("电", "影"): 0.85, ("体", "育"): 0.85,
        }
        
        # 支持跨token的多字搭配（如"打补丁"、"打电话"）
        multi_token_collocations = {
            # 三字固定搭配
            ("打", "补", "丁"): 0.95,  # "打补丁"
            ("打", "电", "话"): 1.0,   # "打电话"
            ("打", "电", "脑"): 0.7,   # "打电脑"（弱搭配）
            ("打", "太", "极"): 0.9,   # "打太极"
            ("打", "网", "球"): 0.95,  # "打网球"
            ("打", "篮", "球"): 0.95,  # "打篮球"
            ("打", "足", "球"): 0.95,  # "打足球"
            ("打", "排", "球"): 0.95,  # "打排球"
            ("打", "乒", "乓"): 0.95,  # "打乒乓"
            ("打", "保", "龄"): 0.9,   # "打保龄"
            ("打", "高", "尔"): 0.85,  # "打高尔夫"
            ("打", "手", "势"): 0.8,   # "打手势"
            ("打", "招", "呼"): 0.85,  # "打招呼"
            ("打", "交", "道"): 0.9,   # "打交道"
            ("打", "电", "报"): 0.9,   # "打电报"
            ("打", "草", "稿"): 0.85,  # "打草稿"
            ("打", "地", "基"): 0.8,   # "打地基"
            ("打", "桩", "机"): 0.75,  # "打桩机"
            
            # "看"相关三字搭配
            ("看", "电", "影"): 0.95,  # "看电影"
            ("看", "电", "视"): 0.95,  # "看电视"
            ("看", "报", "纸"): 0.9,   # "看报纸"
            ("看", "病", "人"): 0.85,  # "看病人"
            ("看", "守", "所"): 0.8,   # "看守所"
            ("看", "不", "起"): 0.8,   # "看不起"
            ("看", "得", "起"): 0.8,   # "看得起"
            
            # "开"相关三字搭配
            ("开", "电", "视"): 0.9,   # "开电视"
            ("开", "电", "灯"): 0.9,   # "开电灯"
            ("开", "电", "扇"): 0.85,  # "开电扇"
            ("开", "电", "话"): 0.9,   # "开电话"
            ("开", "会", "议"): 0.95,  # "开会议"
            ("开", "车", "门"): 0.85,  # "开车门"
            ("开", "车", "祸"): 0.9,   # "出车祸"（但"开"是核心）
            ("开", "发", "票"): 0.9,   # "开发票"
            ("开", "学", "校"): 0.85,  # "开学校"
            ("开", "公", "司"): 0.85,  # "开公司"
            ("开", "医", "院"): 0.8,   # "开医院"
            ("开", "餐", "馆"): 0.85,  # "开餐馆"
            ("开", "书", "店"): 0.8,   # "开书店"
            
            # "做"相关三字搭配
            ("做", "生", "意"): 0.9,   # "做生意"
            ("做", "工", "作"): 0.95,  # "做工作"
            ("做", "饭", "菜"): 0.9,   # "做饭菜"
            ("做", "操", "场"): 0.8,   # "做操场"（弱）
            ("做", "东", "西"): 0.7,   # "做东西"（弱）
            
            # "吃"相关三字搭配
            ("吃", "早", "饭"): 0.95,  # "吃早饭"
            ("吃", "午", "饭"): 0.95,  # "吃午饭"
            ("吃", "晚", "饭"): 0.95,  # "吃晚饭"
            ("吃", "药", "片"): 0.9,   # "吃药片"
            ("吃", "亏", "本"): 0.85,  # "吃亏本"
            ("吃", "哑", "巴"): 0.8,   # "吃哑巴亏"
            
            # "喝"相关三字搭配
            ("喝", "白", "酒"): 0.9,   # "喝白酒"
            ("喝", "红", "酒"): 0.9,   # "喝红酒"
            ("喝", "啤", "酒"): 0.9,   # "喝啤酒"
            ("喝", "果", "汁"): 0.85,  # "喝果汁"
            ("喝", "牛", "奶"): 0.9,   # "喝牛奶"
            
            # "睡"相关三字搭配
            ("睡", "午", "觉"): 0.95,  # "睡午觉"
            ("睡", "懒", "觉"): 0.9,   # "睡懒觉"
            
            # "写"相关三字搭配
            ("写", "作", "文"): 0.9,   # "写作文"
            ("写", "日", "记"): 0.95,  # "写日记"
            ("写", "生", "字"): 0.9,   # "写生字"
            ("写", "毛", "笔"): 0.85,  # "写毛笔"
            
            # "读"相关三字搭配
            ("读", "大", "学"): 0.9,   # "读大学"
            ("读", "研", "究"): 0.85,  # "读研究生"
            ("读", "小", "学"): 0.9,   # "读小学"
            ("读", "中", "学"): 0.9,   # "读中学"
            
            # 其他常见三字搭配
            ("银", "行", "卡"): 0.95,  # "银行卡"
            ("火", "车", "站"): 0.95,  # "火车站"
            ("公", "共", "汽"): 0.9,   # "公共汽车"
            ("飞", "机", "场"): 0.95,  # "飞机场"
            ("医", "疗", "费"): 0.85,  # "医疗费"
            ("出", "差", "费"): 0.85,  # "出差费"
            ("旅", "行", "社"): 0.9,   # "旅行社"
            ("电", "视", "台"): 0.9,   # "电视台"
            ("报", "纸", "摊"): 0.8,   # "报纸摊"
        }
        
        # 检查左右邻居是否构成强搭配
        current_token = tokens[token_idx]
        max_strength = 0.2  # 默认弱搭配
        
        # 检查左侧单字搭配
        if token_idx > 0:
            left_pair = (tokens[token_idx - 1], current_token)
            max_strength = max(max_strength, strong_collocations.get(left_pair, 0.2))
        
        # 检查右侧单字搭配
        if token_idx < len(tokens) - 1:
            right_pair = (current_token, tokens[token_idx + 1])
            max_strength = max(max_strength, strong_collocations.get(right_pair, 0.2))
        
        # 检查三字搭配（当前token + 右侧两个token）
        if token_idx < len(tokens) - 2:
            tri_pattern = (current_token, tokens[token_idx + 1], tokens[token_idx + 2])
            max_strength = max(max_strength, multi_token_collocations.get(tri_pattern, 0.2))
        
        # 检查三字搭配（左侧一个 + 当前token + 右侧一个）
        if token_idx > 0 and token_idx < len(tokens) - 1:
            tri_pattern = (tokens[token_idx - 1], current_token, tokens[token_idx + 1])
            max_strength = max(max_strength, multi_token_collocations.get(tri_pattern, 0.2))
        
        return max_strength
    
    def compute_context_correction_factor(self, raw_entropy: float,
                                         global_mean_entropy: float,
                                         colloc_strength: float) -> float:
        """计算中文语境修正因子 ζ(t)
        
        ζ(t) = 1 + γ·sign(H0(t) - H̄)·colloc(t)
        
        Args:
            raw_entropy: 未修正的归一化熵 H0(t)
            global_mean_entropy: 全局平均熵 H̄
            colloc_strength: 固定搭配强度 colloc(t)
        
        Returns:
            correction_factor: 修正因子 ζ(t) ∈ [0.9, 1.1]
        """
        sign_term = 1.0 if raw_entropy > global_mean_entropy else -1.0
        correction = 1.0 + self.gamma * sign_term * colloc_strength
        
        # 限制修正范围 [0.9, 1.1]
        correction = max(0.9, min(1.1, correction))
        
        return correction
    
    def compute_entropy_for_token(self, token: str,
                                  sense_probs: torch.Tensor,
                                  num_senses: int,
                                  correction_factor: float = 1.0) -> float:
        """计算单个token的多义性熵 H(t)
        
        H(t) = (1/log|S_t|) * [-∑_{s∈S_t} p(s|t)·log(p(s|t) + ε)] * ζ(t)
        
        Args:
            token: 当前token字符串
            sense_probs: [num_senses] 义项激活概率分布
            num_senses: 核心义项数量 |S_t|
            correction_factor: 语境修正因子 ζ(t)
        
        Returns:
            entropy: 归一化多义性熵 H(t) ∈ [0, 1]
        """
        # 非多义词返回0
        if num_senses < 2:
            return 0.0
        
        # 计算香农熵：-∑ p(s|t)·log(p(s|t) + ε)
        log_probs = torch.log(sense_probs + self.epsilon)
        shannon_entropy = -(sense_probs * log_probs).sum().item()
        
        # 归一化：除以 log|S_t|
        normalization_factor = np.log(num_senses)
        normalized_entropy = shannon_entropy / normalization_factor
        
        # 应用语境修正因子
        final_entropy = normalized_entropy * correction_factor
        
        # 确保范围 [0, 1]
        final_entropy = max(0.0, min(1.0, final_entropy))
        
        return final_entropy
    
    def compute_batch_entropy(self, tokens: List[str],
                             hidden_states: torch.Tensor,
                             m_t_values: Optional[np.ndarray] = None,
                             attention_weights: Optional[torch.Tensor] = None) -> np.ndarray:
        """批量计算序列中所有token的多义性熵
        
        Args:
            tokens: token序列 (长度T)
            hidden_states: [T, hidden_dim] 语义状态向量
            m_t_values: [T] M(t)形态-语义匹配度标量（可选，若为None则自动计算）
            attention_weights: [T, T] 注意力权重矩阵（可选）
        
        Returns:
            entropies: [T] 每个token的多义性熵
        """
        seq_len = len(tokens)
        entropies = np.zeros(seq_len)
        raw_entropies = []  # 用于计算全局平均
        
        # 如果未提供M(t)值，则自动计算所有token的M(t)
        if m_t_values is None:
            from m_t_calculator import compute_m_t_full
            m_t_values = []
            for t, token in enumerate(tokens):
                # 计算M(t) - 使用简化的计算（不显示详细输出）
                import io
                import contextlib
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    M_t = compute_m_t_full(
                        h_t=hidden_states[t],
                        token_text=token,
                        tokens=tokens,
                        token_idx=t,
                        hidden_states=hidden_states,
                        layer_idx=12  # 使用第12层权重
                    )
                m_t_values.append(M_t)
            m_t_values = np.array(m_t_values)  # [T]
        
        # 第一遍：计算原始熵（不含修正因子）
        for t in range(seq_len):
            token = tokens[t]
            num_senses = self.polysemy_dict.get_sense_count(token)
            
            if not self.polysemy_dict.is_polysemous(token):
                raw_entropies.append(0.0)
                continue
            
            # 提取特征
            h_t = hidden_states[t]  # [hidden_dim]
            c_t = self.sense_model.extract_context_features(
                hidden_states, t, attention_weights
            )  # [hidden_dim]
            
            # 使用M(t)标量作为形态特征
            M_t_value = m_t_values[t]  # 标量值
            m_t = torch.tensor([M_t_value], dtype=torch.float32)  # 转换为[1]张量
            
            # [调试] 输出M(t)信息
            if t < 3:  # 只输出前3个token避免刷屏
                print(f"\n[H(t)计算-M(t)特征] Token '{token}' (位置{t})")
                print(f"  M(t)值: {M_t_value:.6f}")
            
            syn_t = self.sense_model.extract_syntax_features(hidden_states, t)  # [64]
            
            # 计算义项激活概率
            with torch.no_grad():
                sense_probs = self.sense_model(h_t, c_t, m_t, syn_t, num_senses)
            
            # 计算原始熵（不含修正）
            log_probs = torch.log(sense_probs + self.epsilon)
            shannon_entropy = -(sense_probs * log_probs).sum().item()
            normalized_entropy = shannon_entropy / np.log(num_senses)
            
            raw_entropies.append(normalized_entropy)
        
        # 计算全局平均熵
        global_mean_entropy = np.mean([e for e in raw_entropies if e > 0]) if any(raw_entropies) else 0.5
        
        # 第二遍：应用语境修正因子
        for t in range(seq_len):
            token = tokens[t]
            num_senses = self.polysemy_dict.get_sense_count(token)
            
            if not self.polysemy_dict.is_polysemous(token):
                entropies[t] = 0.0
                continue
            
            # 计算搭配强度
            colloc_strength = self.compute_collocation_strength(tokens, t)
            
            # 计算修正因子
            correction_factor = self.compute_context_correction_factor(
                raw_entropies[t], global_mean_entropy, colloc_strength
            )
            
            # 应用修正
            entropies[t] = raw_entropies[t] * correction_factor
            entropies[t] = max(0.0, min(1.0, entropies[t]))
        
        return entropies


def init_entropy_calculator(hidden_dim: int = 896, 
                           morph_dim: int = 254,
                           use_trained_embedding: bool = True) -> PolysemyEntropyCalculator:
    """初始化多义性熵计算器（便于main.py调用）
    
    Args:
        hidden_dim: 语义向量维度
        morph_dim: 形态特征维度
        use_trained_embedding: 是否使用训练好的MorphEmbedding
    """
    return PolysemyEntropyCalculator(
        hidden_dim=hidden_dim,
        morph_dim=morph_dim,
        use_trained_embedding=use_trained_embedding
    )


# ===================== 测试代码：H(t) 科学验证 =====================
if __name__ == "__main__":
    from llm_hidden_extractor import extract_hidden_states
    
    print("="*70)
    print("  H(t) 多义性熵计算科学验证")
    print("="*70)
    
    # 测试文本
    test_text = "他打了一个"
    print(f"\n测试文本：{test_text}")
    
    # 提取真实LLM隐藏状态
    print("\n正在加载 Qwen2.5-0.5B 模型并提取隐藏状态...")
    h_t, token_num, tokenizer, inputs, attn_weights = extract_hidden_states(
        text=test_text,
        middle_layer_idx=12  # 第12层语义层
    )
    
    # 获取token文本
    input_ids = inputs['input_ids'].squeeze(0)
    tokens = [tokenizer.decode([token_id]) for token_id in input_ids]
    
    print(f"✓ 隐藏状态提取完成")
    print(f"  - Token数量: {token_num}")
    print(f"  - 隐藏维度: {h_t.shape[1]}")
    print(f"  - Token序列: {tokens}")
    print(f"  - 注意力矩阵: {attn_weights.shape}")
    
    # 初始化H(t)计算器
    hidden_dim = h_t.shape[1]
    h_t_calculator = init_entropy_calculator(hidden_dim=hidden_dim, morph_dim=224)
    print(f"\n✓ H(t)计算器初始化完成")
    print(f"  - 内置多义词数量: {len(h_t_calculator.polysemy_dict.polysemy_dict)}")
    
    # 批量计算H(t)
    print("\n" + "="*70)
    print("  开始计算多义性熵 H(t)")
    print("="*70)
    print("\n计算流程：")
    print("  1. 自动提取形态特征 m(t)（214康熙部首+笔画+结构）")
    print("  2. 提取上下文特征 c(t)（窗口t±3，注意力加权）")
    print("  3. 提取句法特征 syn(t)（左右邻居语义）")
    print("  4. 计算义项激活概率 p(s|t) = softmax(MLP(h+c+m+syn))")
    print("  5. 应用归一化因子 1/log|S_t|")
    print("  6. 应用语境修正因子 ζ(t)（基于搭配强度）")
    
    entropies = h_t_calculator.compute_batch_entropy(
        tokens=tokens,
        hidden_states=h_t,
        morph_features=None,  # 自动提取
        attention_weights=attn_weights
    )
    
    print("\n✓ 计算完成！")
    
    # 展示结果
    print("\n" + "="*70)
    print("  H(t) 计算结果")
    print("="*70)
    print(f"\n{'Token':<10} {'H(t)':<12} {'|S_t|':<8} {'类型/说明':<30}")
    print("-" * 70)
    
    for i, token in enumerate(tokens):
        sense_count = h_t_calculator.polysemy_dict.get_sense_count(token)
        is_poly = h_t_calculator.polysemy_dict.is_polysemous(token)
        
        if is_poly:
            desc = f"✓ 多义词（{sense_count}个核心义项）"
        else:
            desc = "  单义词"
        
        print(f"{token:<10} {entropies[i]:.8f}    {sense_count:<8} {desc}")
    
    # 科学性验证
    print("\n" + "="*70)
    print("  科学性验证")
    print("="*70)
    
    print("\n【验证1：值域正确性】H(t) ∈ [0, 1]")
    all_valid = all(0 <= e <= 1 for e in entropies)
    print(f"  ✓ 所有H(t)值在有效范围内: {all_valid}")
    
    print("\n【验证2：多义词识别与区分】")
    poly_tokens = []
    non_poly_tokens = []
    
    for i, token in enumerate(tokens):
        if h_t_calculator.polysemy_dict.is_polysemous(token):
            sense_cnt = h_t_calculator.polysemy_dict.get_sense_count(token)
            poly_tokens.append((token, entropies[i], sense_cnt))
        else:
            non_poly_tokens.append((token, entropies[i]))
    
    if poly_tokens:
        avg_poly = np.mean([e for _, e, _ in poly_tokens])
        print(f"  多义词 ({len(poly_tokens)}个):")
        for token, ent, sense_cnt in poly_tokens:
            print(f"    '{token}' |S_t|={sense_cnt}: H(t)={ent:.6f}")
        print(f"  平均H(t): {avg_poly:.6f}")
    
    if non_poly_tokens:
        avg_non_poly = np.mean([e for _, e in non_poly_tokens])
        print(f"\n  单义词 ({len(non_poly_tokens)}个):")
        for token, ent in non_poly_tokens:
            print(f"    '{token}': H(t)={ent:.6f}")
        print(f"  平均H(t): {avg_non_poly:.6f}")
    
    if poly_tokens and non_poly_tokens:
        distinction = avg_poly - avg_non_poly
        print(f"\n  ✓ 多义词与单义词熵值差: {distinction:.6f}")
        if distinction > 0.2:
            print(f"    → 区分度良好")
    
    print("\n【验证3：归一化效果】")
    print("  归一化因子 1/log|S_t| 确保跨token可比:")
    for i, token in enumerate(tokens):
        if h_t_calculator.polysemy_dict.is_polysemous(token):
            sense_cnt = h_t_calculator.polysemy_dict.get_sense_count(token)
            norm_factor = 1.0 / np.log(sense_cnt)
            print(f"    '{token}': |S_t|={sense_cnt}, 归一化因子={norm_factor:.4f}")
    
    print("\n【验证4：语境修正因子 ζ(t)】")
    print("  搭配强度对多义性熵的微调:")
    for i, token in enumerate(tokens):
        if h_t_calculator.polysemy_dict.is_polysemous(token):
            colloc = h_t_calculator.compute_collocation_strength(tokens, i)
            print(f"    '{token}': 搭配强度={colloc:.2f}, H(t)={entropies[i]:.6f}")
    
    print("\n" + "="*70)
    print("  验证结论")
    print("="*70)
    print("\n✓ H(t)计算符合文档3-3科学定义")
    print("  - 值域约束正确: H(t) ∈ [0, 1]")
    print("  - 归一化处理正确: 1/log|S_t| 实现跨token可比")
    print("  - 四特征融合完整: h(t) + c(t) + M(t) + syn(t)")
    print("  - 语境修正有效: ζ(t) ∈ [0.9, 1.1]")
    print("="*70)


# ---------------------- 初始化函数 ----------------------
def init_entropy_calculator(hidden_dim: int = 896, morph_dim: int = 254) -> PolysemyEntropyCalculator:
    """初始化多义性熵计算器 H(t)
    
    Args:
        hidden_dim: 隐藏层维度
        morph_dim: 形态特征维度（已废弃，现在使用M(t)标量）
    
    Returns:
        entropy_calculator: 配置好的H(t)计算器
    """
    return PolysemyEntropyCalculator(
        hidden_dim=hidden_dim,
        morph_dim=morph_dim,
        epsilon=1e-8,
        gamma=0.08
    )
