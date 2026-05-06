"""
Inference script for substitution cipher decryption.

Implements:
- Greedy decoding
- Beam search decoding
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
from data.generate_data import generate_key, encrypt
from model.transformer import CryptoTransformer
from evaluation.metrics import calculate_all_metrics

# Set random seed
random.seed(42)
torch.manual_seed(42)

# Paths
CHECKPOINT_PATH = Path("/kaggle/working/checkpoints/best_model.pt")

# Model hyperparameters
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 4
D_FF = 512
MAX_LEN = 202
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
        json.dump([{'plaintext': 'test', 'ciphertext': 'test'}], f)
        temp_path = f.name

    dataset = CipherDataset(temp_path, max_len=MAX_LEN)
    os.unlink(temp_path)

    # Create model
    model = CryptoTransformer(
        vocab_size=dataset.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from checkpoint (epoch {checkpoint['epoch']})")
    print(f"Device: {device}\n")


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


def greedy_decode(ciphertext):
    """
    Decrypt ciphertext using greedy decoding.

    Args:
        ciphertext: Encrypted text string

    Returns:
        Decrypted plaintext string
    """
    # Preprocess
    ciphertext = preprocess_text(ciphertext)

    if not ciphertext:
        return ""

    # Truncate if too long
    if len(ciphertext) > 200:
        ciphertext = ciphertext[:200]

    # Encode input
    src_tokens = dataset.encode(ciphertext) + [dataset.EOS_IDX]

    # Pad to max_len
    src_tokens = src_tokens[:MAX_LEN]
    src_tokens += [dataset.PAD_IDX] * (MAX_LEN - len(src_tokens))

    # Convert to tensor
    src = torch.tensor([src_tokens], dtype=torch.long, device=device)

    # Create source mask
    src_mask = (src != dataset.PAD_IDX).unsqueeze(1).unsqueeze(2)

    # Encode
    encoder_output = model.encode(src, src_mask)

    # Decode autoregressively
    tgt = torch.full((1, 1), dataset.SOS_IDX, dtype=torch.long, device=device)

    for _ in range(MAX_LEN - 1):
        # Create causal mask
        tgt_mask = torch.tril(torch.ones(tgt.size(1), tgt.size(1), device=device)).bool()
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

        # Decode
        logits = model(src, tgt, src_mask, tgt_mask)

        # Get next token
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Append
        tgt = torch.cat([tgt, next_token], dim=1)

        # Stop at EOS
        if next_token.item() == dataset.EOS_IDX:
            break

    # Decode to text
    plaintext = dataset.decode(tgt[0])

    return plaintext


def beam_search_decode(ciphertext, beam_size=5):
    """
    Decrypt ciphertext using beam search.

    Args:
        ciphertext: Encrypted text string
        beam_size: Beam width

    Returns:
        Decrypted plaintext string
    """
    # Preprocess
    ciphertext = preprocess_text(ciphertext)

    if not ciphertext:
        return ""

    # Truncate if too long
    if len(ciphertext) > 200:
        ciphertext = ciphertext[:200]

    # Encode input
    src_tokens = dataset.encode(ciphertext) + [dataset.EOS_IDX]
    src_tokens = src_tokens[:MAX_LEN]
    src_tokens += [dataset.PAD_IDX] * (MAX_LEN - len(src_tokens))

    src = torch.tensor([src_tokens], dtype=torch.long, device=device)
    src_mask = (src != dataset.PAD_IDX).unsqueeze(1).unsqueeze(2)

    # Encode
    encoder_output = model.encode(src, src_mask)

    # Initialize beam
    # Each beam element: (sequence, score)
    beams = [(torch.full((1, 1), dataset.SOS_IDX, dtype=torch.long, device=device), 0.0)]

    for _ in range(MAX_LEN - 1):
        new_beams = []

        for seq, score in beams:
            # Stop if sequence ends with EOS
            if seq[0, -1].item() == dataset.EOS_IDX:
                new_beams.append((seq, score))
                continue

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

        # Stop if all beams end with EOS
        if all(seq[0, -1].item() == dataset.EOS_IDX for seq, _ in beams):
            break

    # Return best beam
    best_seq = beams[0][0]
    plaintext = dataset.decode(best_seq[0])

    return plaintext


def decrypt(ciphertext, method='greedy', beam_size=5):
    """
    Main decryption function.

    Args:
        ciphertext: Encrypted text string
        method: 'greedy' or 'beam'
        beam_size: Beam size for beam search

    Returns:
        Decrypted plaintext string
    """
    if method == 'greedy':
        return greedy_decode(ciphertext)
    elif method == 'beam':
        return beam_search_decode(ciphertext, beam_size)
    else:
        raise ValueError(f"Unknown method: {method}")


def run_demo_examples():
    """
    Run hardcoded demo examples.
    """
    print("="*80)
    print("DEMO EXAMPLES")
    print("="*80)

    # Generate demo examples with known plaintexts
    demo_texts = [
        # Typical cases
        "the transformer learns to decrypt substitution ciphers by analyzing patterns in encrypted text",
        "deep learning models can solve complex cryptographic problems with sufficient training data",
        "attention mechanisms allow the model to focus on relevant parts of the input sequence",

        # Edge cases - short text
        "hello world",
        "ai",

        # Edge cases - repeated letters and rare letters
        "the quick brown fox jumps over the lazy dog",
        "zzzz xxxx qqqq",

        # Long sequences
        "in recent years deep learning has revolutionized many fields of artificial intelligence including natural language processing computer vision and speech recognition the transformer architecture introduced",

        "cryptanalysis is the study of analyzing information systems in order to study the hidden aspects of the systems cryptanalysis is used to breach cryptographic security systems and gain access",

        # Failure cases - very short / edge
        "x",
    ]

    for i, plaintext in enumerate(demo_texts, 1):
        # Generate random cipher key
        key = generate_key()
        ciphertext = encrypt(plaintext, key)

        # Decrypt
        predicted = decrypt(ciphertext, method='greedy')

        # Calculate metrics
        metrics = calculate_all_metrics(predicted, plaintext)

        # Determine category
        if i <= 3:
            category = "Typical"
        elif i <= 7:
            category = "Edge Case"
        elif i <= 9:
            category = "Long"
        else:
            category = "Challenge"

        print(f"\n{'='*80}")
        print(f"Example {i} [{category}] - Character Accuracy: {metrics['character_accuracy']:.2f}%")
        print(f"{'='*80}")
        print(f"Ciphertext: {ciphertext[:100]}{'...' if len(ciphertext) > 100 else ''}")
        print(f"Predicted:  {predicted[:100]}{'...' if len(predicted) > 100 else ''}")
        print(f"True:       {plaintext[:100]}{'...' if len(plaintext) > 100 else ''}")
        print(f"BLEU Score: {metrics['bleu_score']:.4f}")


def interactive_cli():
    """
    Interactive command-line interface for decryption.
    """
    print("\n" + "="*80)
    print("INTERACTIVE DECRYPTION CLI")
    print("="*80)
    print("Enter encrypted text to decrypt (or 'quit' to exit)")
    print("The model expects lowercase a-z and spaces only")
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
            plaintext = decrypt(ciphertext, method='greedy')

            print(f"\nDecrypted text: {plaintext}")

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
    print("SUBSTITUTION CIPHER DECRYPTION")
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
