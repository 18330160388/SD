import torch
import numpy as np
import torch.nn as nn
from typing import List, Optional
from scipy.spatial.distance import cosine

# ---------------------- 核心配置（基于文档2定义） ----------------------
# 1. 214个康熙部首（严格按文档要求）
RADICALS = [
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
    "龜", "龠"
]

# 简化字偏旁到康熙部首的映射
SIMPLIFIED_TO_RADICAL = {
    "艹": "艸",  # 草字头
    "氵": "水",  # 三点水
    "忄": "心",  # 竖心旁
    "扌": "手",  # 提手旁
    "饣": "食",  # 食字旁
    "纟": "糸",  # 绞丝旁
    "钅": "金",  # 金字旁
    "衤": "衣",  # 衣字旁
    "礻": "示",  # 示字旁
    "犭": "犬",  # 反犬旁
    "牜": "牛",  # 牛字旁
    "罒": "网",  # 四字头
    "覀": "襾",  # 西字头
    "辶": "辵",  # 走之底
    "廴": "廴",  # 建之底
    "肀": "聿",  # 聿字旁
    "灬": "火"   # 四点底
}

RADICAL_TO_IDX = {rad: i for i, rad in enumerate(RADICALS)}
NUM_RADICALS = len(RADICALS)  # 214（严格符合文档）

# 2. 笔画顺序类别（5类）
STROKE_ORDER = ["横", "竖", "撇", "捺", "折"]
NUM_STROKE_ORDER = len(STROKE_ORDER)  # 5

# 3. 结构类型（4类）
STRUCT_TYPES = ["左右", "上下", "包围", "独体"]
NUM_STRUCT = len(STRUCT_TYPES)  # 4

# 4. 形态特征总维度：214（部首）+ 6（笔画：1+5）+ 4（结构）= 224（严格符合文档定义）
MORPH_DIM = NUM_RADICALS + 1 + NUM_STROKE_ORDER + NUM_STRUCT  # 224

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
        """使用cnradical库获取部首，并映射到214康熙部首"""
        if not self.has_cnradical or self.radical_tool is None:
            return None
        try:
            radical = self.radical_tool.trans_ch(char)
            if not radical:
                return None
            # 简化字偏旁映射到康熙部首
            if radical in SIMPLIFIED_TO_RADICAL:
                radical = SIMPLIFIED_TO_RADICAL[radical]
            return radical
        except Exception:
            return None
    
    def _normalize_radical(self, radical: str) -> str:
        """标准化部首：简化字偏旁转换为康熙部首"""
        if radical in SIMPLIFIED_TO_RADICAL:
            return SIMPLIFIED_TO_RADICAL[radical]
        return radical
    
    def extract(self, char: str) -> Optional[np.ndarray]:
        """
        提取单个/多个汉字的形态特征向量m(t)（严格符合文档定义）
        返回：shape=(224,)的numpy数组
        m(t) = [m_rad(t); m_stroke(t); m_struct(t)]
        - m_rad(t): 214维（部首one-hot）
        - m_stroke(t): 6维（笔画数1维 + 笔画顺序5维）
        - m_struct(t): 4维（结构类型one-hot）
        
        多字token处理策略：取第一个汉字的形态特征（首字通常承载主要语义）
        """
        # 处理多字token：取第一个字符
        if len(char) > 1:
            # 找到第一个汉字
            first_char = None
            for c in char:
                if '\u4e00' <= c <= '\u9fff':
                    first_char = c
                    break
            if first_char is None:
                return None  # 没有汉字
            char = first_char
        
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
        
        # 2. 部首特征（one-hot编码，214维）
        # 先标准化部首（简化字偏旁转换为康熙部首）
        radical = self._normalize_radical(morph_info["radical"])
        radical_onehot = np.zeros(NUM_RADICALS)
        radical_idx = RADICAL_TO_IDX.get(radical, -1)
        if radical_idx != -1:
            radical_onehot[radical_idx] = 1.0
        else:
            # 如果部首不在214个康熙部首中，使用默认值
            print(f"Warning: Radical '{morph_info['radical']}' not in 214 Kangxi radicals for char '{char}'")
        
        # 3. 笔画特征（6维）
        # 3.1 笔画数特征（归一化到[0,1]，1维）
        stroke_count = morph_info["stroke_count"]
        normalized_stroke = min(stroke_count / 30, 1.0)  # 假设最大笔画数30
        
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
        
        # 5. 拼接所有特征（严格符合文档2定义的m(t)结构）
        # m(t) = [m_rad(t); m_stroke(t); m_struct(t)]
        # 维度：214 + (1+5) + 4 = 224
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
        default[NUM_RADICALS] = 0.5  # 笔画数归一化后默认0.5
        default[NUM_RADICALS + 1 + 2] = 1.0  # 笔画顺序默认"撇"
        default[NUM_RADICALS + 1 + NUM_STROKE_ORDER + 3] = 1.0  # 结构默认"独体"
        return default

# ---------------------- 形态特征嵌入函数Φ(m(t))（文档2定义） ----------------------
class MorphEmbedding(nn.Module):
    """将224维形态特征映射至d维语义空间"""
    def __init__(self, morph_dim: int = MORPH_DIM, hidden_dim: int = 896):
        super().__init__()
        # 文档2定义：Linear + GELU + LayerNorm
        self.embedding = nn.Sequential(
            nn.Linear(morph_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, m_t: np.ndarray) -> torch.Tensor:
        """输入：224维形态特征向量；输出：d维形态嵌入向量"""
        # 获取模型所在设备
        device = next(self.embedding.parameters()).device
        m_tensor = torch.tensor(m_t, dtype=torch.float32).unsqueeze(0).to(device)
        return self.embedding(m_tensor).squeeze(0)

# ---------------------- 多义性熵计算（文档2定义的H(t)） ----------------------
def compute_polysemy_entropy(h_t: torch.Tensor, poly_mlp: nn.Module, num_senses: int = 5) -> float:
    """
    计算多义性熵H(t)（即文档2中的V_poly）
    熵越高，义项越模糊；熵越低，义项越明确
    """
    # 确保h_t在模型所在设备上
    device = next(poly_mlp.parameters()).device
    h_t = h_t.to(device)
    sense_logits = poly_mlp(h_t.unsqueeze(0)).squeeze(0)
    sense_probs = torch.softmax(sense_logits, dim=0)
    sense_probs = torch.clamp(sense_probs, min=1e-8, max=1.0)  # 避免log(0)
    entropy = -torch.sum(sense_probs * torch.log(sense_probs)).item()
    return entropy

# ---------------------- M(t)核心计算函数（文档2定义） ----------------------
def compute_m_t(
    h_t: torch.Tensor,
    token_text: str,
    morph_extractor: ChineseMorphExtractor,
    morph_embedding: MorphEmbedding,
    poly_mlp: nn.Module,
    beta: float = 0.2  # 上下文修正因子权重（文档2典型值）
) -> float:
    """
    计算形态-语义匹配度M(t)（文档2核心公式：M(t) = cosine(h(t), Φ(m(t))) · η(t)）
    """
    print(f"\n{'='*60}")
    print(f"[M(t) 计算] Token: '{token_text}' (长度: {len(token_text)})")
    print(f"{'='*60}")
    
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
    print(f"  [OK] 形态特征维度: {m_t.shape[0]} (期望224)")
    
    # 2. 形态特征嵌入Φ(m(t))（映射至语义空间）
    phi_m_t = morph_embedding(m_t).to(h_t.device)
    print(f"  [OK] 形态嵌入维度: {phi_m_t.shape[0]} (语义空间维度)")
    
    # 3. 计算基础匹配度（余弦相似度，截断负相关值）
    h_t_np = h_t.detach().cpu().numpy() if h_t.requires_grad else h_t.cpu().numpy()
    phi_m_t_np = phi_m_t.detach().cpu().numpy() if phi_m_t.requires_grad else phi_m_t.cpu().numpy()
    cos_sim = 1 - cosine(h_t_np, phi_m_t_np)  # 余弦相似度（1-cosine距离）
    base_match = max(cos_sim, 0.0)  # 截断负相关，文档2定义
    
    print(f"\n  【基础匹配度计算】")
    print(f"    原始余弦相似度: {cos_sim:.6f}")
    print(f"    截断后base(t):  {base_match:.6f} (max(cos_sim, 0))")
    
    # 4. 计算上下文动态修正因子η(t) = 1 - β·H(t)
    # 文档定义：η(t) ∈ [0.8, 1.0]
    # H(t)为多义性熵，范围[0, log(S)]，S=5时max≈1.609
    h_t_entropy = compute_polysemy_entropy(h_t, poly_mlp)
    eta_t_raw = 1 - beta * h_t_entropy
    eta_t = np.clip(eta_t_raw, 0.8, 1.0)  # 文档明确要求η(t) ∈ [0.8, 1.0]
    
    print(f"\n  【上下文动态修正】")
    print(f"    多义性熵H(t):   {h_t_entropy:.6f} (范围[0, log(5)≈1.609])")
    print(f"    权重系数β:      {beta:.2f}")
    print(f"    原始η(t):       {eta_t_raw:.6f} (1 - β·H(t))")
    print(f"    截断后η(t):     {eta_t:.6f} (clip到[0.8, 1.0] - 文档要求)")
    
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
    """批量提取所有token的形态特征向量m(t)（224维）"""
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

# ---------------------- 初始化工具（方便外部调用） ----------------------
def init_m_t_tools(hidden_dim: int = 896) -> tuple:
    """
    初始化M(t)计算所需的工具类
    返回：(形态提取器, 形态嵌入模型, 多义性分类器)
    """
    morph_extractor = ChineseMorphExtractor()
    morph_embedding = MorphEmbedding(hidden_dim=hidden_dim)
    # 初始化多义性分类器（文档2定义的poly_mlp）
    poly_mlp = nn.Sequential(
        nn.Linear(hidden_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 5)  # 5个默认义项
    ).eval()
    return morph_extractor, morph_embedding, poly_mlp

# ---------------------- 测试代码（单独运行验证） ----------------------
if __name__ == "__main__":
    # 模拟真实LLM隐藏状态（Qwen2.5-0.5B，896维）
    hidden_dim = 896
    test_h_t = torch.randn(hidden_dim)  # 单个token的语义向量h(t)
    test_token_text = "涛"  # 单字token（形态-语义强关联）
    
    # 初始化工具
    morph_extractor, morph_embedding, poly_mlp = init_m_t_tools(hidden_dim)
    
    # 计算M(t)
    M_t = compute_m_t(
        h_t=test_h_t,
        token_text=test_token_text,
        morph_extractor=morph_extractor,
        morph_embedding=morph_embedding,
        poly_mlp=poly_mlp
    )
    
    print(f"测试token：{test_token_text}")
    print(f"形态-语义匹配度M(t)：{M_t:.6f}")