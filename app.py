"""
Streamlit GUI for Crypto Transformer - Substitution Cipher Cryptanalysis
"""

import sys
import os
import re
import json
import random
import string
import tempfile
from pathlib import Path

import streamlit as st
import torch

REPO_DIR = Path(__file__).parent
sys.path.insert(0, str(REPO_DIR))

CHECKPOINT_PATH = REPO_DIR / "checkpoints" / "best_model.pt"
CURVES_PATH     = REPO_DIR / "checkpoints" / "training_curves.png"

st.set_page_config(
    page_title="Crypto Transformer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .cipher-box {
        background: #1e293b !important;
        color: #7dd3fc !important;
        font-family: monospace;
        font-size: 1rem;
        padding: 1rem;
        border-radius: 8px;
        word-break: break-all;
        line-height: 1.8;
        border: 1px solid #334155;
    }
    .plain-box {
        background: #14532d !important;
        color: #bbf7d0 !important;
        font-family: monospace;
        font-size: 1rem;
        padding: 1rem;
        border-radius: 8px;
        word-break: break-all;
        line-height: 1.8;
        border: 1px solid #166534;
    }
    .key-box {
        background: #312e81 !important;
        color: #fde68a !important;
        font-family: monospace;
        font-size: 1.1rem;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        letter-spacing: 0.15em;
        text-align: center;
        border: 1px solid #4338ca;
    }
    .metric-card {
        border: 2px solid #6366f1;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6366f1;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.7;
    }
    .flow-box {
        background: #1e3a5f !important;
        color: #e2e8f0 !important;
        border: 2px solid #3b82f6 !important;
        border-radius: 10px;
        padding: 1rem 0.5rem;
        text-align: center;
        font-weight: 600;
        font-size: 0.85rem;
        height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .flow-arrow {
        color: #94a3b8 !important;
        font-size: 1.4rem;
        text-align: center;
        padding-top: 1.5rem;
    }
    .arch-block {
        border-left: 4px solid #6366f1;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        background: rgba(99,102,241,0.1) !important;
        color: inherit !important;
    }
    .arch-block-green {
        border-left: 4px solid #22c55e;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        background: rgba(34,197,94,0.1) !important;
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Utility functions -------------------------------------------------------

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def generate_key() -> dict:
    alphabet = list(string.ascii_lowercase)
    shuffled = alphabet.copy()
    random.shuffle(shuffled)
    return {alphabet[i]: shuffled[i] for i in range(26)}


def encrypt(text: str, key: dict) -> str:
    return ''.join(key.get(c, c) for c in text)


def key_to_string(key: dict) -> str:
    inverse = {v: k for k, v in key.items()}
    return ''.join(inverse.get(chr(ord('a') + i), chr(ord('a') + i)) for i in range(26))


def apply_key(ciphertext: str, key_string: str) -> str:
    if len(key_string) != 26:
        return ""
    dm = {chr(ord('a') + i): key_string[i] for i in range(26)}
    return ''.join(dm.get(c, c) for c in ciphertext)


def char_accuracy(pred: str, true: str) -> float:
    if not true:
        return 100.0
    n = max(len(pred), len(true))
    correct = sum(1 for i in range(min(len(pred), len(true))) if pred[i] == true[i])
    return 100.0 * correct / n


def word_accuracy(pred: str, true: str) -> float:
    pw, tw = pred.split(), true.split()
    if not tw:
        return 100.0
    correct = sum(1 for i in range(min(len(pw), len(tw))) if pw[i] == tw[i])
    return 100.0 * correct / len(tw)


def key_accuracy(pred_key: str, true_key: str) -> float:
    if len(pred_key) != 26 or len(true_key) != 26:
        return 0.0
    return 100.0 * sum(1 for a, b in zip(pred_key, true_key) if a == b) / 26


# --- Model loading -----------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def load_model_cached():
    if not CHECKPOINT_PATH.exists():
        return None, None, None

    from data.dataset import CipherDataset
    from model.transformer import CryptoTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([{'plaintext': 'test' + ' ' * 196}], f)
        tmp = f.name

    dataset = CipherDataset(tmp, max_src_len=201)
    os.unlink(tmp)

    model = CryptoTransformer(
        vocab_size=dataset.vocab_size,
        d_model=256,
        num_heads=4,
        num_layers=4,
        d_ff=1024,
        max_len=201,
        dropout=0.1,
    ).to(device)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return model, dataset, device


def predict_key(model, dataset, device, ciphertext: str) -> str:
    MAX_SRC_LEN = 201
    ct = clean_text(ciphertext)
    ct = (ct + ' ' * 200)[:200]

    src_tokens = dataset.encode(ct) + [dataset.EOS_IDX]
    src_tokens = (src_tokens + [dataset.PAD_IDX] * MAX_SRC_LEN)[:MAX_SRC_LEN]

    src = torch.tensor([src_tokens], dtype=torch.long, device=device)
    src_mask = (src != dataset.PAD_IDX).unsqueeze(1).unsqueeze(2)
    tgt = torch.full((1, 1), dataset.SOS_IDX, dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(26):
            tgt_mask = (
                torch.tril(torch.ones(tgt.size(1), tgt.size(1), device=device))
                .bool().unsqueeze(0).unsqueeze(0)
            )
            logits = model(src, tgt, src_mask, tgt_mask)
            tgt = torch.cat([tgt, logits[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1)

    key_str = dataset.decode(tgt[0])
    return (key_str + 'a' * 26)[:26]


# --- Sidebar -----------------------------------------------------------------

st.sidebar.markdown("## Crypto Transformer")
st.sidebar.markdown("Substitution Cipher Cryptanalysis using a Transformer neural network.")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Live Demo", "Architecture", "Training Curves", "About"],
    label_visibility="collapsed",
)

model_obj, dataset_obj, device_obj = load_model_cached()
model_loaded = model_obj is not None

if model_loaded:
    st.sidebar.success("Model loaded")
    ckpt_info = torch.load(CHECKPOINT_PATH, map_location="cpu")
    st.sidebar.caption(f"Epoch: {ckpt_info.get('epoch', '?')}  |  Device: {device_obj}")
else:
    st.sidebar.warning("No checkpoint found. Train the model to enable decryption.")


# =============================================================================
# PAGE: Live Demo
# =============================================================================
if page == "Live Demo":
    st.title("Live Demo")
    st.caption("Enter plaintext, encrypt it with a random substitution cipher, then run the model to decrypt.")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### Input")
        plaintext_input = st.text_area(
            "Plaintext (letters and spaces only):",
            value="the transformer learns to decrypt substitution ciphers by analyzing patterns in the text",
            height=120,
        )
        btn_encrypt = st.button("Encrypt", use_container_width=True, type="primary")

    if "encrypted_result" not in st.session_state:
        st.session_state.encrypted_result = None

    if btn_encrypt:
        cleaned = clean_text(plaintext_input)
        if not cleaned:
            st.error("Please enter some text (letters and spaces only).")
        else:
            key_dict = generate_key()
            st.session_state.encrypted_result = {
                "plaintext": cleaned,
                "ciphertext": encrypt(cleaned, key_dict),
                "true_key": key_to_string(key_dict),
            }

    if st.session_state.encrypted_result:
        res        = st.session_state.encrypted_result
        cleaned    = res["plaintext"]
        ciphertext = res["ciphertext"]
        true_key   = res["true_key"]

        with col_left:
            st.markdown("#### Cipher Key (ground truth)")
            st.markdown(f'<div class="key-box">{true_key}</div>', unsafe_allow_html=True)
            st.caption("26 letters — position i is the plaintext letter that cipher letter (a+i) maps to")

        with col_right:
            st.markdown("#### Plaintext")
            st.markdown(f'<div class="plain-box">{cleaned}</div>', unsafe_allow_html=True)
            st.markdown("#### Ciphertext")
            st.markdown(f'<div class="cipher-box">{ciphertext}</div>', unsafe_allow_html=True)

        st.markdown("---")

        if not model_loaded:
            st.warning(
                "No trained model found. "
                "Run `python training/train.py` then restart the app to enable decryption."
            )
        else:
            if st.button("Decrypt with Transformer", use_container_width=True, type="secondary"):
                with st.spinner("Running inference..."):
                    pred_key   = predict_key(model_obj, dataset_obj, device_obj, ciphertext)
                    pred_plain = apply_key(ciphertext, pred_key)

                ca = char_accuracy(pred_plain.strip(), cleaned.strip())
                wa = word_accuracy(pred_plain.strip(), cleaned.strip())
                ka = key_accuracy(pred_key, true_key)

                st.markdown("#### Results")
                m1, m2, m3 = st.columns(3)
                for col, value, label in [
                    (m1, f"{ca:.1f}%", "Character Accuracy"),
                    (m2, f"{wa:.1f}%", "Word Accuracy"),
                    (m3, f"{ka:.1f}%", "Key Accuracy"),
                ]:
                    with col:
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<div class="metric-value">{value}</div>'
                            f'<div class="metric-label">{label}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                st.markdown("#### Predicted Key")
                key_html = "".join(
                    f'<span style="color:#86efac">{pc}</span>'
                    if pc == tc else
                    f'<span style="color:#fca5a5">{pc}</span>'
                    for pc, tc in zip(pred_key, true_key)
                )
                st.markdown(f'<div class="key-box">{key_html}</div>', unsafe_allow_html=True)
                st.caption("Green = correct letter    Red = wrong letter")

                st.markdown("#### Decrypted Output vs. Ground Truth")
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.markdown("**Model output**")
                    st.markdown(f'<div class="plain-box">{pred_plain.strip()}</div>', unsafe_allow_html=True)
                with dc2:
                    st.markdown("**Ground truth**")
                    st.markdown(f'<div class="plain-box">{cleaned}</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE: Architecture
# =============================================================================
elif page == "Architecture":
    st.title("Model Architecture")
    st.caption('Encoder-Decoder Transformer built from scratch — Vaswani et al., "Attention is All You Need", 2017')

    st.markdown("### Data Flow")
    flow_items = [
        ("Ciphertext", "200 chars"),
        ("",           ""),
        ("Encoder",    "4 layers"),
        ("",           ""),
        ("Cross-Attention", ""),
        ("",           ""),
        ("Decoder",    "4 layers"),
        ("",           ""),
        ("Cipher Key", "26 letters"),
    ]
    flow_cols = st.columns(9)
    for col, (label, sub) in zip(flow_cols, flow_items):
        with col:
            if label == "":
                st.markdown(
                    '<div class="flow-arrow">&#8594;</div>',
                    unsafe_allow_html=True,
                )
            else:
                text = f"{label}<br><small>{sub}</small>" if sub else label
                st.markdown(
                    f'<div class="flow-box">{text}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    col_enc, col_dec = st.columns(2, gap="large")

    with col_enc:
        st.markdown("### Encoder")
        encoder_blocks = [
            ("Character Embeddings",          "vocab_size=30  →  d_model=256"),
            ("Sinusoidal Positional Encoding", "max_len=201,  freq 1/10000^(2i/d)"),
            ("x 4 Encoder Layers",            ""),
            ("    Multi-Head Self-Attention",  "num_heads=4,  d_k=64"),
            ("    Feed-Forward Network",       "256 → 1024 → 256,  ReLU"),
            ("    Layer Norm + Residual",      ""),
        ]
        for title, detail in encoder_blocks:
            indent = title.startswith("    ")
            css    = "arch-block"
            ml     = "margin-left:1.5rem;" if indent else ""
            st.markdown(
                f'<div class="{css}" style="{ml}">'
                f'<strong>{title.strip()}</strong>'
                f'{"<br><small>" + detail + "</small>" if detail else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col_dec:
        st.markdown("### Decoder")
        decoder_blocks = [
            ("Character Embeddings",          "vocab_size=30  →  d_model=256"),
            ("Sinusoidal Positional Encoding", "max_len=28  (SOS + 26 letters + EOS)"),
            ("x 4 Decoder Layers",            ""),
            ("    Masked Self-Attention",      "Causal mask,  num_heads=4"),
            ("    Cross-Attention",            "Attends over encoder output"),
            ("    Feed-Forward Network",       "256 → 1024 → 256,  ReLU"),
            ("    Layer Norm + Residual",      ""),
            ("Linear Projection + Softmax",   "d_model → vocab_size=30"),
        ]
        for title, detail in decoder_blocks:
            indent = title.startswith("    ")
            css    = "arch-block-green"
            ml     = "margin-left:1.5rem;" if indent else ""
            st.markdown(
                f'<div class="{css}" style="{ml}">'
                f'<strong>{title.strip()}</strong>'
                f'{"<br><small>" + detail + "</small>" if detail else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### Hyperparameters")

    params = {
        "d_model": 256,
        "num_heads": 4,
        "num_layers": 4,
        "d_ff": 1024,
        "dropout": 0.1,
        "vocab_size": 30,
        "max_src_len": 201,
        "max_tgt_len": 28,
        "label_smoothing": 0.1,
        "warmup_steps": 4000,
    }
    p_cols = st.columns(5)
    for i, (k, v) in enumerate(params.items()):
        with p_cols[i % 5]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="font-size:1.4rem">{v}</div>'
                f'<div class="metric-label">{k}</div>'
                f'</div><br>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### Parameter Count")
    if model_loaded:
        total_params = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
        st.metric("Total Trainable Parameters", f"{total_params:,}")
    else:
        st.metric("Total Trainable Parameters", "~5.6M (estimated)")

    st.markdown("### Learning Rate Schedule")
    st.latex(r"lr = d_{\text{model}}^{-0.5} \cdot \min\!\left(step^{-0.5},\ step \cdot warmup\_steps^{-1.5}\right)")
    st.caption("Warmup steps = 4,000. LR ramps up linearly then decays as the inverse square root of the step number.")

    st.markdown("### Key Design Decisions")
    decisions = [
        (
            "Character-level tokenization",
            "The cipher operates at the character level, so a character vocabulary naturally captures "
            "letter frequency patterns — no subword tokenization is needed.",
        ),
        (
            "Key prediction instead of direct decryption",
            "Predicting the 26-letter key and applying it deterministically is more structured "
            "than generating 200 plaintext characters autoregressively. It reduces the output "
            "sequence length from 200 to 26 tokens.",
        ),
        (
            "Fresh random keys per training sample",
            "Every training example uses a unique random key, forcing the model to learn general "
            "English language patterns rather than memorizing any specific cipher mapping.",
        ),
        (
            "Teacher forcing during training",
            "The decoder is given the true key prefix at each step during training, enabling "
            "stable and fast convergence. Greedy decoding is used at inference time.",
        ),
    ]
    for title, body in decisions:
        with st.expander(title):
            st.write(body)


# =============================================================================
# PAGE: Training Curves
# =============================================================================
elif page == "Training Curves":
    st.title("Training Curves")
    st.caption("Loss and accuracy over training epochs")

    if CURVES_PATH.exists():
        st.image(str(CURVES_PATH), use_container_width=True)
    else:
        st.info(
            "No training curves image found.\n\n"
            "Run `python training/train.py` — the script saves "
            "`checkpoints/training_curves.png` automatically after training."
        )

    if model_loaded:
        ckpt_info = torch.load(CHECKPOINT_PATH, map_location="cpu")
        st.markdown("---")
        st.markdown("### Checkpoint Info")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Epoch", ckpt_info.get("epoch", "?"))
        with c2:
            val_loss = ckpt_info.get("val_loss", None)
            st.metric("Best Validation Loss", f"{val_loss:.4f}" if val_loss else "?")
        with c3:
            total_params = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
            st.metric("Parameters", f"{total_params:,}")

    st.markdown("---")
    st.markdown("### Expected Results")
    import pandas as pd
    st.dataframe(
        pd.DataFrame({
            "Metric": ["Character Accuracy", "Word Accuracy", "Key Accuracy", "Perfect Key Recovery"],
            "Target":  ["95%+",              "85%+",          "90%+",         "60%+"],
            "Description": [
                "Percentage of characters correctly decrypted",
                "Percentage of words fully correct",
                "Percentage of the 26 key letters predicted correctly",
                "Fraction of samples where all 26 key letters are correct",
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# PAGE: About
# =============================================================================
elif page == "About":
    st.title("About This Project")
    st.caption("Transformer-Based Substitution Cipher Cryptanalysis")

    st.markdown("""
### What is a substitution cipher?

A **substitution cipher** replaces each letter with another letter consistently using a
fixed mapping called the *key*. For example:

| | Text |
|---|---|
| Plaintext | `hello world` |
| Key mapping | h→m, e→j, l→q, o→t, ... |
| Ciphertext | `mjqqt btsqa` |

### What does this model do?

Given only the **ciphertext**, the model predicts the **26-letter decryption key** and
applies it deterministically to recover the original plaintext.

The core challenge is that **every training sample uses a completely fresh random key**,
so the model cannot memorize any specific mapping. It must learn general patterns of the
English language — letter frequencies, common bigrams, word structure — to infer the key
from context alone.

### Training Pipeline

| Step | Description |
|------|-------------|
| Data generation | 300,000 text chunks from WikiText-103, each encrypted with a unique random key |
| Training | Encoder reads ciphertext; decoder generates the 26-letter key with teacher forcing |
| Inference | Greedy decoding predicts the key; the key is applied to decrypt the ciphertext |
| Evaluation | Key accuracy, character accuracy, word accuracy |

### References

- Vaswani et al., *Attention is All You Need*, NeurIPS 2017
- Merity et al., *Pointer Sentinel Mixture Models (WikiText-103)*, ICLR 2017
""")

    st.markdown("---")
    st.markdown("### Project Structure")
    st.code("""\
crypto_transformer/
├── app.py                  GUI (this file)
├── data/
│   ├── generate_data.py    WikiText-103 download and cipher generation
│   └── dataset.py          PyTorch Dataset class
├── model/
│   ├── transformer.py      Encoder-Decoder Transformer
│   └── attention.py        Multi-Head Attention
├── training/
│   ├── train.py            Training loop with teacher forcing
│   └── scheduler.py        Warmup learning rate schedule
├── evaluation/
│   ├── evaluate.py         Test-set evaluation
│   └── metrics.py          Accuracy metrics
├── inference/
│   └── decrypt.py          Greedy and beam search decoding
└── checkpoints/
    └── best_model.pt       Saved after training
""")
