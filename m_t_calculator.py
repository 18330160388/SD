import torch
import numpy as np
import torch.nn as nn
from typing import List, Optional
from scipy.spatial.distance import cosine

# ---------------------- 核心配置（基于文档2定义） ----------------------
# 1. 扩展部首列表：214康熙部首 + 常见简化形式（避免映射转换）
RADICALS = [
    # 214个康熙部首
    # 1画 (6个)
    "一", "丨", "丶", "丿", "乙", "亅",
    # 2画 (29个)
    "二", "亠", "人", "儿", "入", "八", "冂", "冖", "冫", "几", "凵", "刀", "力", "勹", "匕", 
    "匚", "匸", "十", "卜", "卩", "厂", "厶", "又", "口", "囗", "土", "士", "夂", "夊",
    # 3画 (37个)  
    "夕", "大", "女", "子", "宀", "寸", "小", "尢", "尸", "屮", "山", "巛", "工", "己", "巾",
    "干", "幺", "广", "廴", "廾", "弋", "弓", "彐", "彡", "彳", "心", "戈", "戶", "手", "支",
    "攴", "文", "斗", "斤", "方", "无", "日",
    # 4画 (33个)
    "曰", "月", "木", "欠", "止", "歹", "殳", "毋", "比", "毛", "氏", "气", "水", "火", "爪",
    "父", "爻", "爿", "片", "牙", "牛", "犬", "玄", "玉", "瓜", "瓦", "甘", "生", "用", "田",
    "疋", "疒", "癶",
    # 5画 (17个)
    "白", "皮", "皿", "目", "矛", "矢", "石", "示", "禸", "禾", "穴", "立", "竹", "米", "糸",
    "缶", "网",
    # 6画 (19个)
    "羊", "羽", "老", "而", "耒", "耳", "聿", "肉", "臣", "自", "至", "臼", "舌", "舛", "舟",
    "艮", "色", "艸", "虍",
    # 7画 (20个)
    "虫", "血", "行", "衣", "襾", "見", "角", "言", "谷", "豆", "豕", "豸", "貝", "赤", "走",
    "足", "身", "車", "辛", "辰",
    # 8画 (11个)
    "辵", "邑", "酉", "釆", "里", "金", "長", "門", "阜", "隶", "隹",
    # 9画 (10个)
    "雨", "靑", "非", "面", "革", "韋", "韭", "音", "頁", "風",
    # 10画 (6个)
    "飛", "食", "首", "香", "馬", "骨",
    # 11画 (5个)
    "高", "髟", "鬥", "鬯", "鬲",
    # 12画 (5个)
    "鬼", "魚", "鳥", "鹵", "鹿",
    # 13画 (4个)
    "麥", "麻", "黃", "黍",
    # 14画 (4个)
    "黑", "黹", "黽", "鼎",
    # 15画 (2个)
    "鼓", "鼠",
    # 16画 (2个)
    "鼻", "齊",
    # 17画 (2个)
    "齒", "龍",
    # 其余 (2个)
    "龜", "龠",
    
    # ========== 常见简化形式（cnradical常用输出） ==========
    "亻", "氵", "扌", "讠", "钅", "纟", "饣", "衤", "礻", "犭",  # 10个
    "牜", "罒", "覀", "灬", "忄", "阝", "艹", "辶", "⻊", "⺮",  # 10个
    "攵", "⻏", "⺶", "⻗", "⻖", "⺧", "⻞", "⻢", "⻋", "⻉"   # 10个
]

RADICAL_TO_IDX = {rad: i for i, rad in enumerate(RADICALS)}
NUM_RADICALS = len(RADICALS)  # 244（214康熙 + 30简化形式）

# 2. 笔画顺序类别（5类）
STROKE_ORDER = ["横", "竖", "撇", "捺", "折"]
NUM_STROKE_ORDER = len(STROKE_ORDER)  # 5

# 3. 结构类型（4类）
STRUCT_TYPES = ["左右", "上下", "包围", "独体"]
NUM_STRUCT = len(STRUCT_TYPES)  # 4

# 4. 形态特征总维度：244（扩展部首）+ 6（笔画：1+5）+ 4（结构）= 254

MORPH_DIM = NUM_RADICALS + 1 + NUM_STROKE_ORDER + NUM_STRUCT  # 254

# Global singletons for tool management
_GLOBAL_MORPH_EMBEDDING = None
_GLOBAL_MORPH_EXTRACTOR = None
_GLOBAL_H_T_CALCULATOR = None

# ---------------------- 中文形态特征提取工具（基于文档2） ----------------------
class ChineseMorphExtractor:
    """提取中文单字的形态特征：部首、笔画数、笔画顺序、结构类型"""
    def __init__(self):
        # 加载常用汉字的形态信息（可替换为真实字典，此处用示例数据）
        self.char_morph_dict = self._load_char_morph_data()
    
    def _load_char_morph_data(self) -> dict:
        """加载汉字-形态映射（可使用cnradical库自动获取）"""
        try:
            from cnradical import Radical, RunOption
            self.radical_tool = Radical(RunOption.Radical)
            self.has_cnradical = True
        except ImportError:
            self.radical_tool = None
            self.has_cnradical = False
            print("Warning: cnradical not available, using manual dictionary")
        
        # 手工字典作为后备（常用字）
        return {
            "水": {"radical": "水", "stroke_count": 4, "stroke_order": "横", "struct": "独体"},
            "河": {"radical": "氵", "stroke_count": 8, "stroke_order": "横", "struct": "左右"},
            "湖": {"radical": "氵", "stroke_count": 12, "stroke_order": "横", "struct": "左右"},
            "江": {"radical": "氵", "stroke_count": 6, "stroke_order": "横", "struct": "左右"},
            "海": {"radical": "氵", "stroke_count": 10, "stroke_order": "横", "struct": "左右"},
            "松": {"radical": "木", "stroke_count": 8, "stroke_order": "横", "struct": "左右"},
            "柏": {"radical": "木", "stroke_count": 9, "stroke_order": "横", "struct": "左右"},
            "柳": {"radical": "木", "stroke_count": 9, "stroke_order": "横", "struct": "左右"},
            "树": {"radical": "木", "stroke_count": 9, "stroke_order": "横", "struct": "左右"},
            "林": {"radical": "木", "stroke_count": 8, "stroke_order": "横", "struct": "左右"},
            "打": {"radical": "扌", "stroke_count": 5, "stroke_order": "横", "struct": "左右"},
            "拍": {"radical": "扌", "stroke_count": 8, "stroke_order": "横", "struct": "左右"},
            "提": {"radical": "扌", "stroke_count": 12, "stroke_order": "横", "struct": "左右"},
            "抓": {"radical": "扌", "stroke_count": 7, "stroke_order": "横", "struct": "左右"},
            "吃": {"radical": "口", "stroke_count": 6, "stroke_order": "竖", "struct": "左右"},
            "喝": {"radical": "口", "stroke_count": 12, "stroke_order": "竖", "struct": "左右"},
            "叫": {"radical": "口", "stroke_count": 5, "stroke_order": "竖", "struct": "左右"},
            "喊": {"radical": "口", "stroke_count": 12, "stroke_order": "竖", "struct": "左右"},
            "火": {"radical": "火", "stroke_count": 4, "stroke_order": "捺", "struct": "独体"},
            "烧": {"radical": "火", "stroke_count": 10, "stroke_order": "捺", "struct": "左右"},
            "烤": {"radical": "火", "stroke_count": 10, "stroke_order": "捺", "struct": "左右"},
            "灯": {"radical": "火", "stroke_count": 6, "stroke_order": "捺", "struct": "左右"},
            "炎": {"radical": "火", "stroke_count": 8, "stroke_order": "捺", "struct": "上下"},
            "山": {"radical": "山", "stroke_count": 3, "stroke_order": "竖", "struct": "独体"},
            "峰": {"radical": "山", "stroke_count": 10, "stroke_order": "竖", "struct": "左右"},
            "岭": {"radical": "山", "stroke_count": 8, "stroke_order": "竖", "struct": "左右"},
            "我": {"radical": "戈", "stroke_count": 7, "stroke_order": "撇", "struct": "独体"},
            "用": {"radical": "用", "stroke_count": 5, "stroke_order": "竖", "struct": "独体"},
            "苹": {"radical": "艹", "stroke_count": 8, "stroke_order": "横", "struct": "上下"},
            "果": {"radical": "木", "stroke_count": 8, "stroke_order": "横", "struct": "独体"},
            "砸": {"radical": "石", "stroke_count": 10, "stroke_order": "横", "struct": "左右"},
            "手": {"radical": "手", "stroke_count": 4, "stroke_order": "撇", "struct": "独体"},
            "机": {"radical": "木", "stroke_count": 6, "stroke_order": "横", "struct": "左右"},
            "心": {"radical": "心", "stroke_count": 4, "stroke_order": "捺", "struct": "独体"},
            "想": {"radical": "心", "stroke_count": 13, "stroke_order": "捺", "struct": "上下"},
            "情": {"radical": "忄", "stroke_count": 11, "stroke_order": "捺", "struct": "左右"},
            "字": {"radical": "子", "stroke_count": 6, "stroke_order": "横", "struct": "上下"},
            "学": {"radical": "子", "stroke_count": 8, "stroke_order": "捺", "struct": "上下"},
            "好": {"radical": "女", "stroke_count": 6, "stroke_order": "撇", "struct": "左右"},
            "她": {"radical": "女", "stroke_count": 6, "stroke_order": "撇", "struct": "左右"},
            "人": {"radical": "人", "stroke_count": 2, "stroke_order": "撇", "struct": "独体"},
            "他": {"radical": "亻", "stroke_count": 5, "stroke_order": "撇", "struct": "左右"},
            "们": {"radical": "亻", "stroke_count": 5, "stroke_order": "撇", "struct": "左右"},
            "国": {"radical": "囗", "stroke_count": 8, "stroke_order": "横", "struct": "包围"},
            "家": {"radical": "宀", "stroke_count": 10, "stroke_order": "捺", "struct": "上下"},
            "天": {"radical": "大", "stroke_count": 4, "stroke_order": "横", "struct": "独体"},
            "地": {"radical": "土", "stroke_count": 6, "stroke_order": "横", "struct": "左右"},
            "日": {"radical": "日", "stroke_count": 4, "stroke_order": "横", "struct": "独体"},
            "月": {"radical": "月", "stroke_count": 4, "stroke_order": "撇", "struct": "独体"},
            "雨": {"radical": "雨", "stroke_count": 8, "stroke_order": "横", "struct": "独体"},
            "雪": {"radical": "雨", "stroke_count": 11, "stroke_order": "横", "struct": "上下"},
            "风": {"radical": "風", "stroke_count": 4, "stroke_order": "撇", "struct": "包围"},
            "云": {"radical": "二", "stroke_count": 4, "stroke_order": "横", "struct": "独体"},
            "电": {"radical": "雨", "stroke_count": 5, "stroke_order": "竖", "struct": "独体"},
            "金": {"radical": "金", "stroke_count": 8, "stroke_order": "撇", "struct": "上下"},
            "银": {"radical": "钅", "stroke_count": 11, "stroke_order": "撇", "struct": "左右"},
            "铁": {"radical": "钅", "stroke_count": 10, "stroke_order": "撇", "struct": "左右"},
            "鱼": {"radical": "魚", "stroke_count": 8, "stroke_order": "撇", "struct": "独体"},
            "鸟": {"radical": "鳥", "stroke_count": 5, "stroke_order": "撇", "struct": "独体"},
            "马": {"radical": "馬", "stroke_count": 3, "stroke_order": "横", "struct": "独体"},
            "牛": {"radical": "牛", "stroke_count": 4, "stroke_order": "撇", "struct": "独体"},
            "羊": {"radical": "羊", "stroke_count": 6, "stroke_order": "点", "struct": "独体"},
            "狗": {"radical": "犭", "stroke_count": 8, "stroke_order": "撇", "struct": "左右"},
            "猫": {"radical": "犭", "stroke_count": 11, "stroke_order": "撇", "struct": "左右"},
            "草": {"radical": "艹", "stroke_count": 9, "stroke_order": "横", "struct": "上下"},
            "花": {"radical": "艹", "stroke_count": 7, "stroke_order": "横", "struct": "上下"},
            "米": {"radical": "米", "stroke_count": 6, "stroke_order": "点", "struct": "独体"},
            "饭": {"radical": "饣", "stroke_count": 7, "stroke_order": "撇", "struct": "左右"},
            "食": {"radical": "食", "stroke_count": 9, "stroke_order": "撇", "struct": "上下"},
            "衣": {"radical": "衣", "stroke_count": 6, "stroke_order": "点", "struct": "独体"},
            "服": {"radical": "月", "stroke_count": 8, "stroke_order": "横", "struct": "左右"},
            "车": {"radical": "車", "stroke_count": 4, "stroke_order": "横", "struct": "独体"},
            "路": {"radical": "足", "stroke_count": 13, "stroke_order": "竖", "struct": "左右"},
            "走": {"radical": "走", "stroke_count": 7, "stroke_order": "横", "struct": "独体"},
            "跑": {"radical": "足", "stroke_count": 12, "stroke_order": "竖", "struct": "左右"},
            "足": {"radical": "足", "stroke_count": 7, "stroke_order": "竖", "struct": "独体"},
            "目": {"radical": "目", "stroke_count": 5, "stroke_order": "竖", "struct": "独体"},
            "看": {"radical": "目", "stroke_count": 9, "stroke_order": "撇", "struct": "上下"},
            "见": {"radical": "見", "stroke_count": 4, "stroke_order": "竖", "struct": "独体"},
            "言": {"radical": "言", "stroke_count": 7, "stroke_order": "点", "struct": "独体"},
            "说": {"radical": "言", "stroke_count": 9, "stroke_order": "点", "struct": "左右"},
            "话": {"radical": "言", "stroke_count": 8, "stroke_order": "点", "struct": "左右"},
            "门": {"radical": "門", "stroke_count": 3, "stroke_order": "点", "struct": "包围"},
            "问": {"radical": "門", "stroke_count": 6, "stroke_order": "点", "struct": "包围"},
            "开": {"radical": "廾", "stroke_count": 4, "stroke_order": "横", "struct": "独体"},
            "关": {"radical": "門", "stroke_count": 6, "stroke_order": "点", "struct": "上下"},
        }
    
    def _get_radical_from_cnradical(self, char: str) -> Optional[str]:
        """使用cnradical库获取部首（直接使用，无需映射）"""
        if not self.has_cnradical or self.radical_tool is None:
            return None
        try:
            radical = self.radical_tool.trans_ch(char)
            return radical if radical else None
        except Exception:
            return None
    
    def extract(self, char: str) -> Optional[np.ndarray]:
        """
        提取单个/多个汉字的形态特征向量m(t)
        返回：shape=(254,)的numpy数组
        m(t) = [m_rad(t); m_stroke(t); m_struct(t)]
        - m_rad(t): 244维（扩展部首one-hot）
        - m_stroke(t): 6维（笔画数1维 + 笔画顺序5维）
        - m_struct(t): 4维（结构类型one-hot）
        
        多字token处理策略：融合所有汉字的形态特征（加权平均）
        """
        # 处理多字token：提取所有汉字并融合特征
        if len(char) > 1:
            chinese_chars = [c for c in char if '\u4e00' <= c <= '\u9fff']
            if not chinese_chars:
                return None  # 没有汉字
            
            # 如果是多字词，融合所有字符的形态特征
            if len(chinese_chars) > 1:
                all_features = []
                for c in chinese_chars:
                    feat = self._extract_single_char(c)
                    if feat is not None:
                        all_features.append(feat)
                
                if not all_features:
                    return self._get_default_morph()
                
                # 加权平均：首字权重更高
                weights = np.array([0.5] + [0.5 / (len(all_features) - 1)] * (len(all_features) - 1))
                m_t = np.average(all_features, axis=0, weights=weights)
                return m_t
            else:
                char = chinese_chars[0]
        
        return self._extract_single_char(char)
    
    def _extract_single_char(self, char: str) -> Optional[np.ndarray]:
        """提取单个汉字的形态特征（内部方法）"""
    def _extract_single_char(self, char: str) -> Optional[np.ndarray]:
        """提取单个汉字的形态特征（内部方法）"""
        # 判断是否为中文字符
        if not ('\u4e00' <= char <= '\u9fff'):
            return None
        
        # 1. 查找形态信息（优先用手工字典，其次用cnradical）
        morph_info = self.char_morph_dict.get(char, None)
        
        if not morph_info:
            # 尝试用cnradical自动获取部首
            radical = self._get_radical_from_cnradical(char)
            if radical:
                # 构建基本形态信息（笔画数等用估算值）
                morph_info = {
                    "radical": radical,
                    "stroke_count": 8,  # 默认值
                    "stroke_order": "横",  # 默认值
                    "struct": "左右"  # 默认值
                }
            else:
                return self._get_default_morph()
        
        # 2. 部首特征（one-hot编码，244维扩展部首）
        radical = morph_info["radical"]
        radical_onehot = np.zeros(NUM_RADICALS)
        radical_idx = RADICAL_TO_IDX.get(radical, -1)
        if radical_idx != -1:
            radical_onehot[radical_idx] = 1.0
        else:
            # 如果部首不在扩展列表中，使用默认值（不再警告）
            pass  # 静默处理未知部首
        
        # 3. 笔画特征（6维）
        # 3.1 笔画数特征（归一化到[0,1]，1维）
        stroke_count = morph_info["stroke_count"]
        normalized_stroke = min(stroke_count / 20, 1.0)  # 归一化基准：常用汉字最大笔画数20
        
        # 3.2 笔画顺序特征（one-hot编码，5维）
        stroke_order_onehot = np.zeros(NUM_STROKE_ORDER)
        stroke_order = morph_info["stroke_order"]
        if stroke_order in STROKE_ORDER:
            stroke_order_idx = STROKE_ORDER.index(stroke_order)
            stroke_order_onehot[stroke_order_idx] = 1.0
        
        # 4. 结构类型特征（one-hot编码，4维）
        struct_onehot = np.zeros(NUM_STRUCT)
        struct_type = morph_info["struct"]
        if struct_type in STRUCT_TYPES:
            struct_idx = STRUCT_TYPES.index(struct_type)
            struct_onehot[struct_idx] = 1.0
        
        # 5. 拼接所有特征
        # m(t) = [m_rad(t); m_stroke(t); m_struct(t)]
        # 维度：244 + (1+5) + 4 = 254
        m_t = np.concatenate([
            radical_onehot,              # 214维
            np.array([normalized_stroke]),  # 1维（笔画数）
            stroke_order_onehot,         # 5维（笔画顺序）
            struct_onehot                # 4维（结构类型）
        ])
        
        assert m_t.shape[0] == MORPH_DIM, f"m(t) dimension mismatch: {m_t.shape[0]} != {MORPH_DIM}"
        return m_t
    
    def _get_default_morph(self) -> np.ndarray:
        """未知汉字的默认形态特征（均匀分布）"""
        default = np.zeros(MORPH_DIM)
        default[NUM_RADICALS] = 0.5  # 笔画数归一化后默认0.5（中等复杂度）
        # 笔画顺序默认"撇"（索引2）
        default[NUM_RADICALS + 1 + STROKE_ORDER.index("撇")] = 1.0
        # 结构类型默认"独体"（索引3）
        default[NUM_RADICALS + 1 + NUM_STROKE_ORDER + STRUCT_TYPES.index("独体")] = 1.0
        return default

# ---------------------- 形态特征嵌入函数Φ(m(t))（文档2定义） ----------------------
class MorphEmbedding(nn.Module):
    """将254维形态特征映射至d维语义空间"""
    def __init__(self, morph_dim: int = MORPH_DIM, hidden_dim: int = 896):
        super().__init__()
        # 文档2定义：Linear + GELU + LayerNorm
        self.embedding = nn.Sequential(
            nn.Linear(morph_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, m_t: np.ndarray) -> torch.Tensor:
        """输入：254维形态特征向量；输出：d维形态嵌入向量"""
        # 获取模型所在设备
        device = next(self.embedding.parameters()).device
        m_tensor = torch.tensor(m_t, dtype=torch.float32).unsqueeze(0).to(device)
        return self.embedding(m_tensor).squeeze(0)

# ---------------------- 多义性熵计算（调用完整实现） ----------------------
# H(t) 计算使用 h_t_calculator.py 中的完整实现
# 这里保留简化接口用于兼容性

def compute_polysemy_entropy_simple(
    h_t: torch.Tensor, 
    poly_mlp: nn.Module, 
    num_senses: int = 5,
    epsilon: float = 1e-8,
    zeta_t: float = 1.0
) -> float:
    """
    简化版多义性熵计算（仅用于快速测试）
    
    ⚠️ 警告：此版本仅使用 h(t) 单一特征，不符合完整规范
    生产环境请使用 compute_polysemy_entropy_full()
    
    公式：H(t) = [1/log|S_t|] * [-Σ p(s|t) * log(p(s|t) + ε)] * ζ(t)
    """
    device = next(poly_mlp.parameters()).device
    h_t = h_t.to(device)
    
    sense_logits = poly_mlp(h_t.unsqueeze(0)).squeeze(0)
    sense_probs = torch.softmax(sense_logits, dim=0)
    
    sense_probs_safe = sense_probs + epsilon
    raw_entropy = -torch.sum(sense_probs * torch.log(sense_probs_safe)).item()
    
    normalization_factor = 1.0 / np.log(num_senses)
    zeta_t = np.clip(zeta_t, 0.9, 1.1)
    
    normalized_entropy = normalization_factor * raw_entropy * zeta_t
    return normalized_entropy

# ---------------------- M(t)核心计算函数（文档2定义） ----------------------
def compute_m_t(
    h_t: torch.Tensor,
    token_text: str,
    beta: float = 0.2
) -> float:
    """
    计算形态-语义匹配度M(t)（文档2核心公式：M(t) = cosine(h(t), Φ(m(t)))）
    """
    global _GLOBAL_MORPH_EXTRACTOR, _GLOBAL_MORPH_EMBEDDING
    if _GLOBAL_MORPH_EXTRACTOR is None:
        _GLOBAL_MORPH_EXTRACTOR = ChineseMorphExtractor()
    if _GLOBAL_MORPH_EMBEDDING is None:
        _GLOBAL_MORPH_EMBEDDING = MorphEmbedding()
    morph_extractor = _GLOBAL_MORPH_EXTRACTOR
    morph_embedding = _GLOBAL_MORPH_EMBEDDING
    poly_mlp = nn.Sequential(
        nn.Linear(896, 128),
        nn.ReLU(),
        nn.Linear(128, 5)
    ).eval()

    # 1. 提取形态特征向量m(t)
    m_t = morph_extractor.extract(token_text)
    if m_t is None:
        M_t_result = 0.3  # 非单字token默认中等匹配度（文档2边界条件）
        print(f"  ❌ 形态特征提取失败（非中文字符）")
        print(f"  ➜ M(t) = {M_t_result:.6f} (默认值)")
        print(f"{'='*60}\n")
        return M_t_result

    if len(token_text) > 1:
        # 简单显示：如果首字是汉字则显示，否则显示(非汉字)
        first_char = token_text[0]
        is_chinese = '\u4e00' <= first_char <= '\u9fff'
        display_char = first_char if is_chinese else '(非汉字)'
        print(f"  ⚠️  多字token，提取首字 '{display_char}' 的形态特征")
    print(f"  [OK] 形态特征维度: {m_t.shape[0]} (期望254)")

    # 2. 形态特征嵌入Φ(m(t))（映射至语义空间）
    phi_m_t = morph_embedding(m_t).to(h_t.device)
    # 推理阶段强制归一化（LayerNorm），确保分布对齐
    norm = torch.nn.LayerNorm(phi_m_t.shape[0]).to(phi_m_t.device)
    phi_m_t = norm(phi_m_t)
    print(f"  [OK] 形态嵌入维度: {phi_m_t.shape[0]} (语义空间维度, 已归一化)")

    # 3. 计算基础匹配度（余弦相似度，截断负相关值）
    h_t_np = h_t.detach().cpu().numpy() if h_t.requires_grad else h_t.cpu().numpy()
    phi_m_t_np = phi_m_t.detach().cpu().numpy() if phi_m_t.requires_grad else phi_m_t.cpu().numpy()
    cos_sim = 1 - cosine(h_t_np, phi_m_t_np)  # 余弦相似度（1-cosine距离）
    base_match = max(cos_sim, 0.0)  # 截断负相关，文档2定义

    print(f"\n  【基础匹配度计算】")
    print(f"    原始余弦相似度: {cos_sim:.6f}")
    print(f"    截断后base(t):  {base_match:.6f} (max(cos_sim, 0))")

    # 4. 计算上下文动态修正因子η(t) = 1 - β·H(t)
    h_t_entropy = compute_polysemy_entropy_simple(h_t, poly_mlp, num_senses=5)
    eta_t_raw = 1 - beta * h_t_entropy
    eta_t = np.clip(eta_t_raw, 0.8, 1.0)  # 确保η(t) ∈ [0.8, 1.0]

    print(f"\n  【上下文动态修正】")
    print(f"    多义性熵H(t):   {h_t_entropy:.6f} (简化版，仅h(t)特征)")
    print(f"    权重系数β:      {beta:.2f} (最大修正幅度)")
    print(f"    原始η(t):       {eta_t_raw:.6f} (1 - {beta} × {h_t_entropy:.6f})")
    print(f"    截断后η(t):     {eta_t:.6f} (clip到[0.8, 1.0])")

    # 5. 最终M(t)计算：M(t) = cosine(h(t), Φ(m(t))) · η(t)
    M_t = base_match * eta_t
    M_t_result = round(M_t, 6)  # 保留6位小数

    print(f"\n  【最终结果】")
    print(f"    M(t) = cosine(h(t), Φ(m(t))) × η(t)")
    print(f"         = {base_match:.6f} × {eta_t:.6f}")
    print(f"         = {M_t_result:.6f}")
    print(f"    [OK] M(t) in [0,1]: {0 <= M_t_result <= 1}")
    print(f"{'='*60}\n")

    return M_t_result

# ---------------------- 批量计算函数 ----------------------
def extract_m_t_batch(
    token_texts: List[str],
    morph_extractor: ChineseMorphExtractor
) -> List[np.ndarray]:
    """批量提取所有token的形态特征向量m(t)（254维）"""
    m_t_list = []
    for token_text in token_texts:
        m_t = morph_extractor.extract(token_text)
        m_t_list.append(m_t)
    return m_t_list

def compute_m_t_batch(
    hidden_states: torch.Tensor,
    token_texts: List[str],
    morph_extractor: ChineseMorphExtractor,
    morph_embedding: MorphEmbedding,
    poly_mlp: nn.Module
) -> np.ndarray:
    """批量计算所有token的M(t)"""
    seq_len = hidden_states.shape[0]
    M_t_list = []
    for token_idx in range(seq_len):
        h_t = hidden_states[token_idx]
        token_text = token_texts[token_idx]
        M_t = compute_m_t(
            h_t=h_t,
            token_text=token_text,
            morph_extractor=morph_extractor,
            morph_embedding=morph_embedding,
            poly_mlp=poly_mlp
        )
        M_t_list.append(M_t)
    return np.array(M_t_list)

# ---------------------- 新的Φ(m(t))：字符级语义向量（无需训练）----------------------
def compute_phi_m_t(token_text: str, model, tokenizer, device) -> torch.Tensor:
    """
    用字符级embedding替代形态嵌入（无需训练W_m）
    
    公式：Φ(m(t)) = LayerNorm(Mean(Embedding(chars(t))))
    
    理论依据：
    1. LLM的embedding层已在大规模语料上训练，"江"的embedding天然接近"水"语义
    2. 字符embedding与h(t)在同一空间，可直接计算相似度
    3. 参考文献：CharacterBERT (ACL 2020), Glyce (ACL 2019)
    
    Args:
        token_text: token文本（如"江"、"金属"）
        model: LLM模型（用于提取embedding）
        tokenizer: 分词器
        device: 设备
        
    Returns:
        Φ(m(t)): d维语义向量（与h(t)同维度）
    """
    # 提取汉字字符
    chars = [c for c in token_text if '\u4e00' <= c <= '\u9fff']
    if not chars:
        return torch.zeros(model.config.hidden_size).to(device)
    
    # 提取每个字符的embedding
    char_embeddings = []
    for char in chars:
        # 编码单个字符
        char_ids = tokenizer.encode(char, add_special_tokens=False)
        if len(char_ids) > 0:
            # 从embedding层提取（这是LLM学习过的语义表示）
            with torch.no_grad():
                embed = model.get_input_embeddings()(
                    torch.tensor([char_ids[0]]).to(device)
                )
            char_embeddings.append(embed.squeeze(0))
    
    if not char_embeddings:
        return torch.zeros(model.config.hidden_size).to(device)
    
    # 多字token取平均
    phi_m_t = torch.stack(char_embeddings).mean(dim=0)
    
    # LayerNorm对齐分布（与文档定义一致）
    phi_m_t = torch.nn.functional.layer_norm(phi_m_t, (phi_m_t.shape[0],))
    
    return phi_m_t


# ---------------------- 完整版M(t)计算（使用h_t_calculator） ----------------------
def compute_m_t_full(
    h_t: torch.Tensor,
    token_text: str,
    tokens: List[str],
    token_idx: int,
    hidden_states: torch.Tensor,
    model=None,  # 保留兼容性（不再使用）
    tokenizer=None,  # 保留兼容性（不再使用）
    attention_weights: Optional[torch.Tensor] = None,
    beta: float = 0.2,
    layer_idx: Optional[int] = None
) -> float:
    """
    完整版M(t)计算，使用训练好的MorphEmbedding

    公式：M(t) = cosine(h(t), Φ(m(t))) × η(H(t))
    其中：Φ(m(t)) = LayerNorm(GELU(W_m·m(t) + b_m))（使用训练好的W_m）

    Args:
        h_t: 当前token的语义向量
        token_text: 当前token文本
        tokens: 完整token序列
        token_idx: 当前token在序列中的索引
        hidden_states: 完整序列的隐藏状态 [seq_len, hidden_dim]
        model: 保留兼容性（废弃）
        tokenizer: 保留兼容性（废弃）
        attention_weights: 注意力权重矩阵（可选）
        beta: 修正因子权重

    Returns:
        M(t): 形态-语义匹配度 ∈ [0, 1]
    """
    print(f"\n{'='*60}")
    print(f"[完整版M(t)计算] Token: '{token_text}' (位置: {token_idx})")
    print(f"{'='*60}")
    
    # 1. 提取当前token形态特征
    global _GLOBAL_MORPH_EXTRACTOR, _GLOBAL_MORPH_EMBEDDING
    # 层数推断：优先从layer_idx参数，否则从hidden_states.shape[-1]推断
    if _GLOBAL_MORPH_EXTRACTOR is None or _GLOBAL_MORPH_EMBEDDING is None:
        # 动态加载权重路径
        from pathlib import Path
        hidden_dim = hidden_states.shape[-1]
        # 使用layer_idx指定层数，默认12层
        layer_folder = str(layer_idx) if layer_idx is not None else "12"
        # 路径：train_morph_embedding/layer_models/layer_{layer}/morph_embedding_best.pt
        model_path = Path(__file__).parent / "train_morph_embedding" / "layer_models" / f"layer_{layer_folder}" / "morph_embedding_best.pt"
        morph_extractor = ChineseMorphExtractor()
        morph_embedding = MorphEmbedding(hidden_dim=hidden_dim)
        if model_path.exists():
            morph_embedding.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"[OK] 已加载训练权重: {model_path}")
        else:
            print(f"[警告] 未找到训练权重: {model_path}，使用随机初始化")
        morph_embedding.eval()
        _GLOBAL_MORPH_EXTRACTOR = morph_extractor
        _GLOBAL_MORPH_EMBEDDING = morph_embedding
    m_t = _GLOBAL_MORPH_EXTRACTOR.extract(token_text)
    if m_t is None:
        print(f"  [X] 非中文token，返回默认值")
        return 0.3
    
    print(f"  [1/4] 原始形态特征 m(t)")
    print(f"    维度: {m_t.shape}")
    print(f"    非零维度数: {np.count_nonzero(m_t)}")
    print(f"    前20维: {m_t[:20]}")
    print(f"    均值: {np.mean(m_t):.6f}, 方差: {np.var(m_t):.6f}, 最大: {np.max(m_t):.6f}, 最小: {np.min(m_t):.6f}")
    print(f"    范数: {np.linalg.norm(m_t):.6f}")
    
    # 2. 计算Φ(m(t))：直接调用MorphEmbedding模型，并对 h(t) 和 Φ(m(t)) 都做 LayerNorm 归一化
    m_t_tensor = torch.from_numpy(m_t).float().unsqueeze(0)
    with torch.no_grad():
        phi_m_t = _GLOBAL_MORPH_EMBEDDING(m_t).cpu()
    # LayerNorm 归一化
    layer_norm = torch.nn.LayerNorm(phi_m_t.shape[-1])
    phi_m_t = layer_norm(phi_m_t)
    h_t = layer_norm(h_t.cpu())
    print(f"  [2/4] 形态嵌入 Φ(m(t)) = MorphEmbedding(m(t)) (直接模型调用)")
    print(f"    Φ(m(t))最终: 维度{phi_m_t.shape}")
    phi_m_t_np_dbg = phi_m_t.detach().numpy()
    print(f"      前20维: {phi_m_t_np_dbg[:20]}")
    print(f"      范数: {np.linalg.norm(phi_m_t_np_dbg):.6f}")
    print(f"      均值: {np.mean(phi_m_t_np_dbg):.6f}, 标准差: {np.std(phi_m_t_np_dbg):.6f}, 最大: {np.max(phi_m_t_np_dbg):.6f}, 最小: {np.min(phi_m_t_np_dbg):.6f}")
    

    # 3. 基础匹配度：cosine(h(t), Φ(m(t)))
    h_t_np = h_t.detach().cpu().numpy() if hasattr(h_t, 'requires_grad') and h_t.requires_grad else h_t.cpu().numpy()
    phi_m_t_np = phi_m_t.detach().cpu().numpy() if hasattr(phi_m_t, 'requires_grad') and phi_m_t.requires_grad else phi_m_t.cpu().numpy()

    # 余弦相似度 = 1 - cosine_distance
    from scipy.spatial.distance import cosine
    base_match = max(1 - cosine(h_t_np, phi_m_t_np), 0.0)

    print(f"  [3/4] 余弦相似度 cosine(h(t), Φ(m(t)))")
    print(f"    h(t)前20维: {h_t_np[:20]}")
    print(f"    h(t)范数: {np.linalg.norm(h_t_np):.6f}, 均值: {np.mean(h_t_np):.6f}, 标准差: {np.std(h_t_np):.6f}, 最大: {np.max(h_t_np):.6f}, 最小: {np.min(h_t_np):.6f}")
    print(f"    Φ(m(t))范数: {np.linalg.norm(phi_m_t_np):.6f}, 均值: {np.mean(phi_m_t_np):.6f}, 标准差: {np.std(phi_m_t_np):.6f}, 最大: {np.max(phi_m_t_np):.6f}, 最小: {np.min(phi_m_t_np):.6f}")
    print(f"    余弦相似度: {base_match:.6f}")
    
    # 4. 最终M(t)计算：M(t) = cosine(h(t), Φ(m(t)))
    M_t_result = round(base_match, 6)  # 保留6位小数

    print(f"\n  【最终结果】")
    print(f"    M(t) = cosine(h(t), Φ(m(t)))")
    print(f"         = {base_match:.6f}")
    print(f"    [OK] M(t) in [0,1]: {0 <= M_t_result <= 1}")
    print(f"{'='*60}\n")

    return M_t_result