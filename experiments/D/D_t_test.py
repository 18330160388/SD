import os
import sys
import numpy as np

# ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from llm_hidden_extractor import extract_hidden_states
from m_t_calculator import compute_m_t
from d_t_calculator import compute_d_t_batch


def run_dt_for_sentences(sentences, layer_idx=12, window_size=2, sim_threshold=0.5):
    for text in sentences:
        print('\n' + '='*60)
        print(f'句子: {text}')
        print('='*60)

        # try to extract hidden states using existing extractor
        try:
            h_t, token_num, tokenizer, inputs, attn = extract_hidden_states(text, middle_layer_idx=layer_idx)
            hidden = h_t  # torch.Tensor (token_num, hidden_dim)
            # decode tokens from tokenizer if possible
            try:
                input_ids = inputs['input_ids'].squeeze(0).tolist()
                tokens = [tokenizer.decode([int(tid)]) for tid in input_ids]
            except Exception:
                # fallback: split by characters
                tokens = [c for c in text]
        except Exception as e:
            print('Warning: extract_hidden_states failed, falling back to synthetic vectors:', e)
            import torch
            torch.manual_seed(0)
            tokens = [c for c in text]
            hidden = torch.randn(len(tokens), 8)

        # compute M(t) per token using m_t_calculator
        m_t_list = []
        for i, tok in enumerate(tokens):
            try:
                M_t = compute_m_t(h_t=hidden[i], token_text=tok)
            except Exception:
                M_t = 0.0
            m_t_list.append(float(M_t))

        # compute D(t) batch with precomputed M(t)
        try:
            D_arr, diagnostics = compute_d_t_batch(hidden, window_size=window_size, sim_threshold=sim_threshold, precomputed_m_t_list=np.array(m_t_list), return_diagnostics=True)
        except Exception as e:
            print('Error computing D(t):', e)
            continue

        # print per-token diagnostics and mean
        for i, tok in enumerate(tokens):
            diag = diagnostics[i]
            D_t = float(D_arr[i])
            M_t = m_t_list[i]
            seff = diag.get('Seff_size')
            fb = diag.get('fallback_used')
            print(f"Token[{i}] ({tok})  D(t)={D_t:.6f}  M(t)={M_t:.6f}  Seff={seff}  fallback={fb}")

        mean_D = float(np.mean(D_arr))
        print(f"平均 D(t) = {mean_D:.6f}")


if __name__ == '__main__':
    sentences = [
        "翻来覆去睡不着",
        "她去公园",
    ]
    run_dt_for_sentences(sentences)
