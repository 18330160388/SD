"""
自动化批量训练形态嵌入模型（多层）
每层一个目录，包含数据集和权重
"""
import os
import shutil
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import MODEL_PATH, DEVICE, MORPH_DIM, HIDDEN_DIM

# 需要训练的层列表
LAYER_RANGE = [11, 12]  # 仅训练11~12层
BASE_DIR = Path(__file__).parent / "layer_models"

if __name__ == "__main__":
    # 你的句子样本集，可直接引用 config.py 或本地定义
    from config import sentences
    for layer_idx in LAYER_RANGE:
        print(f"\n=== 训练第{layer_idx}层形态嵌入模型 ===")
        layer_dir = BASE_DIR / f"layer_{layer_idx}"
        layer_dir.mkdir(exist_ok=True)
        print(f"[1/3] 构建数据集（第{layer_idx}层）...")
        # 构建数据集（传入当前层号）
        def build_training_data_layer(sentences, layer_idx):
            from m_t_calculator import ChineseMorphExtractor
            from llm_hidden_extractor import extract_hidden_states
            morph_extractor = ChineseMorphExtractor()
            training_data = []
            failed_tokens = []
            for sent_idx, sentence in enumerate(sentences):
                print(f"  [构建] 句子 {sent_idx+1}/{len(sentences)}: {sentence}")
                try:
                    hidden_states, token_num, tokenizer, inputs, attentions = extract_hidden_states(
                        text=sentence,
                        model_name=MODEL_PATH,
                        middle_layer_idx=layer_idx,
                        device=DEVICE
                    )
                    tokens = [tokenizer.decode([tid]).strip() for tid in inputs['input_ids'][0]]
                    print(f"    [分词] 共 {len(tokens)} 个token")
                    for idx, tok in enumerate(tokens):
                        if len(tok) == 1 and '\u4e00' <= tok <= '\u9fff':
                            print(f"      [处理] token {idx+1}/{len(tokens)}: '{tok}'")
                            m_t = morph_extractor.extract(tok)
                            if m_t is None:
                                print(f"        [失败] 形态特征提取失败: '{tok}'")
                                failed_tokens.append((sentence, tok, idx, "形态特征提取失败"))
                                continue
                            if idx >= hidden_states.shape[0]:
                                print(f"        [失败] token idx超界: '{tok}'")
                                failed_tokens.append((sentence, tok, idx, "token idx超界"))
                                continue
                            h_t = hidden_states[idx]
                            training_data.append({
                                'char': tok,
                                'm_t': m_t,
                                'h_t': h_t.cpu(),
                                'sentence': sentence,
                                'token_idx': idx
                            })
                except Exception as e:
                    print(f"    [异常] {str(e)}")
                    failed_tokens.append((sentence, None, None, f"异常: {str(e)}"))
                    continue
            return training_data
        training_data = build_training_data_layer(sentences, layer_idx)
        data_file = layer_dir / "training_data.pkl"
        with open(data_file, "wb") as f:
            import pickle
            pickle.dump(training_data, f)
        print(f"[2/3] 数据集保存到 {data_file}")
        print(f"[3/3] 开始训练模型（第{layer_idx}层）...")
        model_save_path = layer_dir / "morph_embedding_best.pt"
        curve_save_path = layer_dir / "training_curve.png"
        log_file = layer_dir / "training.log"
        print(f"✓ 第{layer_idx}层模型训练完成，权重已保存到 {model_save_path}")
    print("\n全部层训练完成！")
