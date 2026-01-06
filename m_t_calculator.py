import torch
import numpy as np
import torch.nn as nn
from typing import List, Optional
from scipy.spatial.distance import cosine

# ---------------------- 核心配置（基于文档2定义） ----------------------
# 1. 214个部首（简化版，完整列表可参考《现代汉语词典》部首表）
RADICALS = [
    "一", "丨", "丿", "丶", "乀", "乙", "二", "十", "厂", "匚", "刂", "卜", "冂", "亻", "八", "人", "入", "勹", "几", "儿",
    "冫", "冖", "凵", "卩", "刀", "力", "勑", "匕", "匚", "匸", "十", "卜", "卩", "厂", "厶", "又", "口", "囗", "土", "士",
    "夂", "夕", "大", "女", "子", "宀", "寸", "小", "尢", "尸", "屮", "山", "巛", "工", "己", "巾", "干", "幺", "广",
    "廴", "廾", "弋", "弓", "彐", "彡", "彳", "心", "戈", "户", "手", "支", "攴", "文", "斗", "斤", "方", "无", "日",
    "曰", "月", "木", "欠", "止", "歹", "殳", "毋", "比", "毛", "氏", "气", "水", "火", "爪", "父", "爻", "爿", "片",
    "牙", "牛", "犬", "玄", "玉", "瓜", "瓦", "甘", "生", "用", "田", "疋", "疒", "癶", "白", "皮", "皿", "目", "矛",
    "矢", "石", "示", "禸", "禾", "穴", "立", "竹", "米", "糸", "缶", "网", "羊", "羽", "老", "而", "耒", "耳", "聿",
    "肉", "臣", "自", "至", "臼", "舌", "舟", "艮", "色", "艸", "虍", "虫", "血", "行", "衣", "西", "见", "角", "言",
    "谷", "豆", "豕", "豸", "贝", "赤", "走", "足", "身", "车", "辛", "辰", "酉", "釆", "里", "金", "长", "门", "隶",
    "隹", "雨", "青", "非", "奄", "龟", "齿", "黾", "麻", "鹿", "麦", "鱼", "鸟", "卤", "鹿", "麻", "黑", "黹", "黾",
    "鼎", "鼓", "鼠", "鼻", "齐", "良", "艸", "艹", "饣", "饣", "忄", "氵", "冫", "灬", "火", "礻", "衤", "钅", "牜",
    "犭", "扌", "纟", "糹", "罒", "衤", "覀", "辶", "廴", "肀", "丿", "丶", "乛", "乙", "乚", "丿", "丶", "乀", "乁"
]
RADICAL_TO_IDX = {rad: i for i, rad in enumerate(RADICALS)}
NUM_RADICALS = len(RADICALS)  # 214

# 2. 笔画顺序类别（5类）
STROKE_ORDER = ["横", "竖", "撇", "捺", "折"]
NUM_STROKE_ORDER = len(STROKE_ORDER)  # 5

# 3. 结构类型（4类）
STRUCT_TYPES = ["左右", "上下", "包围", "独体"]
NUM_STRUCT = len(STRUCT_TYPES)  # 4

# 4. 形态特征总维度：214（部首）+ 1（笔画数归一化）+ 5（笔画顺序）+ 4（结构）= 224（文档2定义）
MORPH_DIM = NUM_RADICALS + 1 + NUM_STROKE_ORDER + NUM_STRUCT  # 224

# ---------------------- 中文形态特征提取工具（基于文档2） ----------------------
class ChineseMorphExtractor:
    """提取中文单字的形态特征：部首、笔画数、笔画顺序、结构类型"""
    def __init__(self):
        # 加载常用汉字的形态信息（可替换为真实字典，此处用示例数据）
        self.char_morph_dict = self._load_char_morph_data()
    
    def _load_char_morph_data(self) -> dict:
        """加载汉字-形态映射（真实场景可从中文NLP工具如LTP、HanLP获取）"""
        return {
            "水": {"radical": "氵", "stroke_count": 4, "stroke_order": "捺", "struct": "独体"},
            "河": {"radical": "氵", "stroke_count": 8, "stroke_order": "横", "struct": "左右"},
            "湖": {"radical": "氵", "stroke_count": 12, "stroke_order": "横", "struct": "左右"},
            "松": {"radical": "木", "stroke_count": 8, "stroke_order": "横", "struct": "左右"},
            "柏": {"radical": "木", "stroke_count": 9, "stroke_order": "横", "struct": "左右"},
            "打": {"radical": "扌", "stroke_count": 5, "stroke_order": "横", "struct": "左右"},
            "拍": {"radical": "扌", "stroke_count": 8, "stroke_order": "横", "struct": "左右"},
            "吃": {"radical": "口", "stroke_count": 6, "stroke_order": "竖", "struct": "左右"},
            "喝": {"radical": "口", "stroke_count": 12, "stroke_order": "竖", "struct": "左右"},
            "火": {"radical": "火", "stroke_count": 4, "stroke_order": "点", "struct": "独体"},
            "烧": {"radical": "火", "stroke_count": 10, "stroke_order": "点", "struct": "左右"},
            "山": {"radical": "山", "stroke_count": 3, "stroke_order": "竖", "struct": "独体"},
            "峰": {"radical": "山", "stroke_count": 10, "stroke_order": "竖", "struct": "左右"},
            "我": {"radical": "戈", "stroke_count": 7, "stroke_order": "撇", "struct": "独体"},
            "用": {"radical": "用", "stroke_count": 5, "stroke_order": "竖", "struct": "独体"},
            "苹": {"radical": "艹", "stroke_count": 8, "stroke_order": "横", "struct": "上下"},
            "果": {"radical": "木", "stroke_count": 8, "stroke_order": "横", "struct": "独体"},
            "砸": {"radical": "石", "stroke_count": 10, "stroke_order": "横", "struct": "左右"},
            "手": {"radical": "手", "stroke_count": 4, "stroke_order": "撇", "struct": "独体"},
            "机": {"radical": "木", "stroke_count": 6, "stroke_order": "横", "struct": "左右"}
        }
    
    def extract(self, char: str) -> Optional[np.ndarray]:
        """
        提取单个汉字的形态特征向量m(t)（文档2定义）
        返回：shape=(224,)的numpy数组
        """
        if len(char) != 1:
            return None  # 仅支持单字token
        
        # 1. 查找形态信息（无则返回默认值）
        morph_info = self.char_morph_dict.get(char, None)
        if not morph_info:
            return self._get_default_morph()
        
        # 2. 部首特征（one-hot编码）
        radical_onehot = np.zeros(NUM_RADICALS)
        radical_idx = RADICAL_TO_IDX.get(morph_info["radical"], -1)
        if radical_idx != -1:
            radical_onehot[radical_idx] = 1.0
        
        # 3. 笔画数特征（归一化到[0,1]，文档2定义）
        stroke_count = morph_info["stroke_count"]
        normalized_stroke = min(stroke_count / 30, 1.0)  # 假设最大笔画数30
        
        # 4. 笔画顺序特征（one-hot编码）
        stroke_order_onehot = np.zeros(NUM_STROKE_ORDER)
        stroke_order = morph_info["stroke_order"]
        if stroke_order in STROKE_ORDER:
            stroke_order_idx = STROKE_ORDER.index(stroke_order)
            stroke_order_onehot[stroke_order_idx] = 1.0
        
        # 5. 结构类型特征（one-hot编码）
        struct_onehot = np.zeros(NUM_STRUCT)
        struct_type = morph_info["struct"]
        if struct_type in STRUCT_TYPES:
            struct_idx = STRUCT_TYPES.index(struct_type)
            struct_onehot[struct_idx] = 1.0
        
        # 拼接所有特征（文档2定义的m(t)结构）
        m_t = np.concatenate([
            radical_onehot,
            np.array([normalized_stroke]),
            stroke_order_onehot,
            struct_onehot
        ])
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
    # 1. 提取形态特征向量m(t)
    m_t = morph_extractor.extract(token_text)
    if m_t is None:
        return 0.3  # 非单字token默认中等匹配度（文档2边界条件）
    
    # 2. 形态特征嵌入Φ(m(t))（映射至语义空间）
    phi_m_t = morph_embedding(m_t).to(h_t.device)
    
    # 3. 计算基础匹配度（余弦相似度，截断负相关值）
    h_t_np = h_t.detach().cpu().numpy() if h_t.requires_grad else h_t.cpu().numpy()
    phi_m_t_np = phi_m_t.detach().cpu().numpy() if phi_m_t.requires_grad else phi_m_t.cpu().numpy()
    cos_sim = 1 - cosine(h_t_np, phi_m_t_np)  # 余弦相似度（1-cosine距离）
    base_match = max(cos_sim, 0.0)  # 截断负相关，文档2定义
    
    # 4. 计算上下文动态修正因子η(t) = 1 - β·H(t)
    h_t_entropy = compute_polysemy_entropy(h_t, poly_mlp)
    eta_t = 1 - beta * h_t_entropy
    eta_t = np.clip(eta_t, 0.8, 1.0)  # 文档2定义η(t)∈[0.8,1.0]
    
    # 5. 最终M(t)计算
    M_t = base_match * eta_t
    return round(M_t, 6)  # 保留6位小数

# ---------------------- 批量计算函数 ----------------------
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
    test_token_text = "水"  # 单字token（形态-语义强关联）
    
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
    print(f"预期结果：因'水'与部首'氵'强关联，M(t)应接近1.0")