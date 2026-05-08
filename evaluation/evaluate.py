"""
Evaluation script for trained substitution cipher Transformer.

Loads the best checkpoint and evaluates on the test set.
Reports all metrics and shows example predictions.
"""

import sys
import os
import torch
import json
import random
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import CipherDataset
from model.transformer import CryptoTransformer
from evaluation.metrics import aggregate_metrics, calculate_all_metrics

# Set random seed
random.seed(42)
torch.manual_seed(42)

# Paths - use relative paths for local execution
SCRIPT_DIR = Path(__file__).parent.parent
DATA_DIR = SCRIPT_DIR / "data" / "generated"
CHECKPOINT_PATH = SCRIPT_DIR / "checkpoints" / "best_model.pt"
RESULTS_DIR = SCRIPT_DIR / "evaluation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model hyperparameters (must match training config)
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 6
D_FF = 2048
MAX_LEN = 202
DROPOUT = 0.1


def create_masks(src, tgt, pad_idx):
    """
    Create attention masks for inference.

    Args:
        src: Source tensor (batch, src_seq_len)
        tgt: Target tensor (batch, tgt_seq_len)
        pad_idx: Index of PAD token

    Returns:
        src_mask: Padding mask for source
        tgt_mask: Causal mask for target
    """
    # Source padding mask
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    # Target causal mask
    tgt_seq_len = tgt.size(1)
    tgt_causal_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len, device=tgt.device)).bool()
    tgt_causal_mask = tgt_causal_mask.unsqueeze(0).unsqueeze(0)

    # Target padding mask
    tgt_padding_mask = (tgt != pad_idx).unsqueeze(1)
    tgt_mask = tgt_causal_mask & tgt_padding_mask.unsqueeze(3)

    return src_mask, tgt_mask


def greedy_decode(model, src, src_mask, dataset, device, max_len=MAX_LEN):
    """
    Greedy decoding for inference.

    Args:
        model: Trained model
        src: Source tensor (batch, src_seq_len)
        src_mask: Source mask
        dataset: Dataset object (for token indices)
        device: Device
        max_len: Maximum decoding length

    Returns:
        Decoded predictions (batch, seq_len)
    """
    model.eval()
    batch_size = src.size(0)

    # Encode source
    encoder_output = model.encode(src, src_mask)

    # Start with SOS token
    tgt = torch.full((batch_size, 1), dataset.SOS_IDX, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        # Create target mask
        tgt_mask = torch.tril(torch.ones(tgt.size(1), tgt.size(1), device=device)).bool()
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        # Decode
        logits = model(src, tgt, src_mask, tgt_mask)

        # Get next token (greedy)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Append to target
        tgt = torch.cat([tgt, next_token], dim=1)

        # Stop if all sequences have EOS
        if (next_token == dataset.EOS_IDX).all():
            break

    return tgt


def evaluate_model(model, dataloader, dataset, device):
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        dataloader: Test data loader
        dataset: Dataset object
        device: Device

    Returns:
        all_predictions: List of predicted strings
        all_targets: List of target strings
        all_ciphertexts: List of ciphertext strings
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_ciphertexts = []

    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating"):
            src = src.to(device)

            # Create source mask
            src_mask = (src != dataset.PAD_IDX).unsqueeze(1).unsqueeze(2)

            # Greedy decode
            predictions = greedy_decode(model, src, src_mask, dataset, device)

            # Decode predictions and targets
            for i in range(src.size(0)):
                pred_text = dataset.decode(predictions[i])
                target_text = dataset.decode(tgt[i])
                cipher_text = dataset.decode(src[i])

                all_predictions.append(pred_text)
                all_targets.append(target_text)
                all_ciphertexts.append(cipher_text)

    return all_predictions, all_targets, all_ciphertexts


def print_examples(predictions, targets, ciphertexts, num_examples=10):
    """
    Print random example predictions.

    Args:
        predictions: List of predicted strings
        targets: List of target strings
        ciphertexts: List of ciphertext strings
        num_examples: Number of examples to print
    """
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS")
    print("="*80)

    # Sample random indices
    indices = random.sample(range(len(predictions)), min(num_examples, len(predictions)))

    for i, idx in enumerate(indices, 1):
        cipher = ciphertexts[idx]
        pred = predictions[idx]
        target = targets[idx]

        # Calculate character accuracy for this example
        char_acc = calculate_all_metrics(pred, target)['character_accuracy']

        print(f"\n--- Example {i} (Char Accuracy: {char_acc:.2f}%) ---")
        print(f"Ciphertext: {cipher[:100]}{'...' if len(cipher) > 100 else ''}")
        print(f"Predicted:  {pred[:100]}{'...' if len(pred) > 100 else ''}")
        print(f"Target:     {target[:100]}{'...' if len(target) > 100 else ''}")


def main():
    """
    Main evaluation function.
    """
    print("="*80)
    print("EVALUATING SUBSTITUTION CIPHER TRANSFORMER")
    print("="*80)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = CipherDataset(DATA_DIR / "test.json", max_len=MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    vocab_size = test_dataset.vocab_size
    print(f"Test samples: {len(test_dataset)}")
    print(f"Vocabulary size: {vocab_size}\n")

    # Load model
    print("Loading model...")
    model = CryptoTransformer(
        vocab_size=vocab_size,
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
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Checkpoint val loss: {checkpoint['val_loss']:.4f}\n")

    # Evaluate
    print("Running evaluation on test set...")
    predictions, targets, ciphertexts = evaluate_model(model, test_loader, test_dataset, device)

    # Calculate metrics
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)

    metrics = aggregate_metrics(predictions, targets)

    print(f"\nCharacter Accuracy: {metrics['character_accuracy']:.2f}%")
    print(f"Word Accuracy:      {metrics['word_accuracy']:.2f}%")
    print(f"BLEU Score:         {metrics['bleu_score']:.4f}")
    print(f"Mean Edit Distance: {metrics['mean_edit_distance']:.2f}")

    # Print examples
    print_examples(predictions, targets, ciphertexts, num_examples=10)

    # Save results
    results = {
        'metrics': metrics,
        'num_samples': len(predictions),
        'checkpoint_epoch': int(checkpoint['epoch']),
        'checkpoint_val_loss': float(checkpoint['val_loss'])
    }

    results_path = RESULTS_DIR / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to {results_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
