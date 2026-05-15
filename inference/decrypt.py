"""
Inference script for substitution cipher KEY prediction and decryption.

NEW TWO-STEP APPROACH:
1. Predict 26-letter substitution key from ciphertext
2. Apply predicted key deterministically to decrypt

Implements:
- Greedy decoding for key prediction
- Beam search decoding for better key prediction
- Deterministic decryption using predicted key
- Interactive CLI
- Demo examples
"""

import sys
import os
import torch
import random
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import CipherDataset
from data.generate_data import generate_key, encrypt, key_to_string
from model.transformer import CryptoTransformer
from evaluation.metrics import calculate_all_metrics, apply_key

# Set random seed
random.seed(42)
torch.manual_seed(42)

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
CHECKPOINT_PATH = SCRIPT_DIR / "checkpoints" / "best_model.pt"

# Model hyperparameters (must match training config)
D_MODEL = 256
NUM_HEADS = 4
NUM_LAYERS = 4
D_FF = 1024
MAX_SRC_LEN = 201
MAX_TGT_LEN = 28
DROPOUT = 0.1
VOCAB_SIZE = 30

# Global model and dataset
model = None
dataset = None
device = None


def load_model():
    """
    Load trained model from checkpoint.
    """
    global model, dataset, device

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dummy dataset to get vocabulary
    import tempfile
    import json

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([{'plaintext': 'test' + ' ' * 196}], f)
        temp_path = f.name

    dataset = CipherDataset(temp_path, max_src_len=MAX_SRC_LEN)
    os.unlink(temp_path)

    # Create model
    model = CryptoTransformer(
        vocab_size=dataset.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len=max(MAX_SRC_LEN, MAX_TGT_LEN),
        dropout=DROPOUT
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from checkpoint (epoch {checkpoint['epoch']})")
    print(f"Device: {device}")
    print(f"Task: Predict 26-letter substitution key\n")


def preprocess_text(text):
    """
    Preprocess input text to match training format.

    Args:
        text: Raw input text

    Returns:
        Cleaned text (lowercase a-z and spaces only)
    """
    import re
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_key_greedy(ciphertext):
    """
    Predict substitution key using greedy decoding.

    Args:
        ciphertext: Encrypted text string (200 chars)

    Returns:
        Predicted 26-letter key string
    """
    # Preprocess
    ciphertext = preprocess_text(ciphertext)

    if not ciphertext:
        return ""

    # Pad or truncate to 200 chars
    if len(ciphertext) < 200:
        ciphertext = ciphertext + ' ' * (200 - len(ciphertext))
    else:
        ciphertext = ciphertext[:200]

    # Encode input
    src_tokens = dataset.encode(ciphertext) + [dataset.EOS_IDX]
    src_tokens = src_tokens[:MAX_SRC_LEN]
    src_tokens += [dataset.PAD_IDX] * (MAX_SRC_LEN - len(src_tokens))

    # Convert to tensor
    src = torch.tensor([src_tokens], dtype=torch.long, device=device)

    # Create source mask
    src_mask = (src != dataset.PAD_IDX).unsqueeze(1).unsqueeze(2)

    # Encode
    encoder_output = model.encode(src, src_mask)

    # Decode key autoregressively (fixed length: 26 letters)
    tgt = torch.full((1, 1), dataset.SOS_IDX, dtype=torch.long, device=device)

    for _ in range(26):  # Predict exactly 26 letters
        # Create causal mask
        tgt_mask = torch.tril(torch.ones(tgt.size(1), tgt.size(1), device=device)).bool()
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

        # Decode
        logits = model(src, tgt, src_mask, tgt_mask)

        # Get next token
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Append
        tgt = torch.cat([tgt, next_token], dim=1)

    # Decode to text (skip SOS, should be exactly 26 letters)
    key_string = dataset.decode(tgt[0])

    # Ensure exactly 26 letters
    if len(key_string) < 26:
        key_string = key_string + 'a' * (26 - len(key_string))
    else:
        key_string = key_string[:26]

    return key_string


def predict_key_beam_search(ciphertext, beam_size=5):
    """
    Predict substitution key using beam search.

    Args:
        ciphertext: Encrypted text string (200 chars)
        beam_size: Beam width

    Returns:
        Predicted 26-letter key string
    """
    # Preprocess
    ciphertext = preprocess_text(ciphertext)

    if not ciphertext:
        return ""

    # Pad or truncate to 200 chars
    if len(ciphertext) < 200:
        ciphertext = ciphertext + ' ' * (200 - len(ciphertext))
    else:
        ciphertext = ciphertext[:200]

    # Encode input
    src_tokens = dataset.encode(ciphertext) + [dataset.EOS_IDX]
    src_tokens = src_tokens[:MAX_SRC_LEN]
    src_tokens += [dataset.PAD_IDX] * (MAX_SRC_LEN - len(src_tokens))

    src = torch.tensor([src_tokens], dtype=torch.long, device=device)
    src_mask = (src != dataset.PAD_IDX).unsqueeze(1).unsqueeze(2)

    # Encode
    encoder_output = model.encode(src, src_mask)

    # Initialize beam
    beams = [(torch.full((1, 1), dataset.SOS_IDX, dtype=torch.long, device=device), 0.0)]

    for step in range(26):  # Predict exactly 26 letters
        new_beams = []

        for seq, score in beams:
            # Create mask
            tgt_mask = torch.tril(torch.ones(seq.size(1), seq.size(1), device=device)).bool()
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

            # Get logits
            logits = model(src, seq, src_mask, tgt_mask)
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

            # Get top-k tokens
            topk_probs, topk_indices = torch.topk(log_probs, beam_size)

            # Create new beams
            for i in range(beam_size):
                new_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, new_token], dim=1)
                new_score = score + topk_probs[0, i].item()
                new_beams.append((new_seq, new_score))

        # Keep top beam_size beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

    # Return best beam
    best_seq = beams[0][0]
    key_string = dataset.decode(best_seq[0])

    # Ensure exactly 26 letters
    if len(key_string) < 26:
        key_string = key_string + 'a' * (26 - len(key_string))
    else:
        key_string = key_string[:26]

    return key_string


def decrypt(ciphertext, method='greedy', beam_size=5):
    """
    TWO-STEP DECRYPTION:
    Step 1: Predict 26-letter substitution key
    Step 2: Apply key deterministically to decrypt ciphertext

    Args:
        ciphertext: Encrypted text string
        method: 'greedy' or 'beam'
        beam_size: Beam size for beam search

    Returns:
        Dictionary with 'key' and 'plaintext'
    """
    # Step 1: Predict key
    if method == 'greedy':
        predicted_key = predict_key_greedy(ciphertext)
    elif method == 'beam':
        predicted_key = predict_key_beam_search(ciphertext, beam_size)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Step 2: Apply key to decrypt
    plaintext = apply_key(ciphertext, predicted_key)

    return {
        'key': predicted_key,
        'plaintext': plaintext
    }


def run_demo_examples():
    """
    Run hardcoded demo examples.
    """
    print("="*80)
    print("DEMO EXAMPLES - KEY PREDICTION APPROACH")
    print("="*80)

    # Generate demo examples with known plaintexts
    demo_texts = [
        "the transformer learns to decrypt substitution ciphers by analyzing patterns in encrypted text" + ' ' * 103,
        "deep learning models can solve complex cryptographic problems with sufficient training data and compute power" + ' ' * 89,
        "attention mechanisms allow the model to focus on relevant parts of the input sequence for better predictions" + ' ' * 90,
        "hello world this is a test of the key prediction system" + ' ' * 144,
        "the quick brown fox jumps over the lazy dog" + ' ' * 155,
    ]

    all_metrics = []

    for i, plaintext in enumerate(demo_texts, 1):
        # Generate random cipher key
        key_dict = generate_key()
        true_key = key_to_string(key_dict)
        ciphertext = encrypt(plaintext, key_dict)

        # Decrypt using model
        result = decrypt(ciphertext, method='greedy')
        predicted_key = result['key']
        predicted_plaintext = result['plaintext']

        # Calculate metrics
        metrics = calculate_all_metrics(predicted_key, true_key, ciphertext, plaintext.strip())

        all_metrics.append(metrics)

        print(f"\n{'='*80}")
        print(f"Example {i}")
        print(f"{'='*80}")
        print(f"Ciphertext:   {ciphertext[:80]}...")
        print(f"True key:     {true_key}")
        print(f"Predicted key: {predicted_key}")
        print(f"Key accuracy:  {metrics['key_accuracy']:.2f}%")
        print(f"Perfect key:   {'✓' if metrics['perfect_key_recovery'] == 1.0 else '✗'}")
        print(f"\nTrue plaintext:      {plaintext.strip()[:80]}...")
        print(f"Predicted plaintext: {predicted_plaintext.strip()[:80]}...")
        print(f"Character accuracy:  {metrics['character_accuracy']:.2f}%")

    # Summary statistics
    from evaluation.metrics import aggregate_metrics
    agg = aggregate_metrics(all_metrics)

    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Average key accuracy:         {agg['key_accuracy']:.2f}%")
    print(f"Perfect key recovery rate:    {agg['perfect_key_recovery_rate']:.2f}%")
    print(f"Average character accuracy:   {agg['character_accuracy']:.2f}%")
    print(f"Average word accuracy:        {agg['word_accuracy']:.2f}%")


def interactive_cli():
    """
    Interactive command-line interface for decryption.
    """
    print("\n" + "="*80)
    print("INTERACTIVE KEY PREDICTION & DECRYPTION CLI")
    print("="*80)
    print("Enter encrypted text to decrypt (or 'quit' to exit)")
    print("The model will predict the substitution key and decrypt")
    print("="*80)

    while True:
        try:
            ciphertext = input("\nEnter encrypted text: ").strip()

            if ciphertext.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break

            if not ciphertext:
                print("Please enter some text.")
                continue

            # Decrypt
            result = decrypt(ciphertext, method='greedy')

            print(f"\nPredicted key: {result['key']}")
            print(f"Decrypted text: {result['plaintext']}")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """
    Main function.
    """
    print("="*80)
    print("SUBSTITUTION CIPHER KEY PREDICTION & DECRYPTION")
    print("="*80)

    # Load model
    print("\nLoading model...")
    load_model()

    # Run demo examples
    run_demo_examples()

    # Interactive CLI (commented out for notebook execution)
    # Uncomment the line below to enable interactive mode
    # interactive_cli()


if __name__ == '__main__':
    main()
