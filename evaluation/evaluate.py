"""
Evaluation script for trained substitution cipher KEY prediction model.

Evaluates model on test set and reports:
- Key accuracy (% of 26 letters correct)
- Perfect key recovery rate (% with all 26 letters perfect)
- Character accuracy (after applying predicted key)
- Word accuracy (after applying predicted key)
"""

import sys
import os
import torch
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import CipherDataset
from model.transformer import CryptoTransformer
from evaluation.metrics import calculate_all_metrics, aggregate_metrics, apply_key

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
DATA_DIR = SCRIPT_DIR / "data" / "generated"
CHECKPOINT_PATH = SCRIPT_DIR / "checkpoints" / "best_model.pt"

# Model hyperparameters (must match training config)
D_MODEL = 256
NUM_HEADS = 4
NUM_LAYERS = 4
D_FF = 1024
MAX_SRC_LEN = 201
MAX_TGT_LEN = 28
DROPOUT = 0.1

# Evaluation settings
NUM_TEST_SAMPLES = 1000  # Evaluate on subset for speed (set to None for full test set)


def predict_key(model, src, dataset, device):
    """
    Predict substitution key from ciphertext.

    Args:
        model: Trained transformer model
        src: Source tensor (batch, max_src_len)
        dataset: Dataset object (for decoding)
        device: Device (cuda or cpu)

    Returns:
        List of predicted key strings (batch_size,)
    """
    model.eval()
    batch_size = src.size(0)

    with torch.no_grad():
        # Create source mask
        src_mask = (src != dataset.PAD_IDX).unsqueeze(1).unsqueeze(2)

        # Encode
        encoder_output = model.encode(src, src_mask)

        # Decode key autoregressively (fixed length: 26 letters)
        tgt = torch.full((batch_size, 1), dataset.SOS_IDX, dtype=torch.long, device=device)

        for _ in range(26):  # Predict exactly 26 letters
            # Create causal mask
            tgt_mask = torch.tril(torch.ones(tgt.size(1), tgt.size(1), device=device)).bool()
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)

            # Decode
            logits = model(src, tgt, src_mask, tgt_mask)

            # Get next token
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Append
            tgt = torch.cat([tgt, next_token], dim=1)

    # Decode to text (batch)
    predicted_keys = []
    for i in range(batch_size):
        key_string = dataset.decode(tgt[i])
        # Ensure exactly 26 letters
        if len(key_string) < 26:
            key_string = key_string + 'a' * (26 - len(key_string))
        else:
            key_string = key_string[:26]
        predicted_keys.append(key_string)

    return predicted_keys


def evaluate_model(model, dataloader, dataset, device):
    """
    Evaluate model on test set.

    Args:
        model: Trained transformer model
        dataloader: Test data loader
        dataset: Dataset object
        device: Device (cuda or cpu)

    Returns:
        Dictionary with aggregate metrics
    """
    model.eval()
    all_metrics = []

    print("\nEvaluating model on test set...")
    print("="*80)

    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating"):
            src = src.to(device)
            tgt = tgt.to(device)

            batch_size = src.size(0)

            # Decode true keys (skip SOS)
            true_keys = []
            for i in range(batch_size):
                key_string = dataset.decode(tgt[i])
                if len(key_string) < 26:
                    key_string = key_string + 'a' * (26 - len(key_string))
                else:
                    key_string = key_string[:26]
                true_keys.append(key_string)

            # Predict keys
            predicted_keys = predict_key(model, src, dataset, device)

            # Evaluate each sample in batch
            for i in range(batch_size):
                # Decode ciphertext
                ciphertext = dataset.decode(src[i])

                # Get keys
                true_key = true_keys[i]
                predicted_key = predicted_keys[i]

                # For true plaintext, decrypt using true key
                true_plaintext = apply_key(ciphertext, true_key)

                # Calculate metrics
                metrics = calculate_all_metrics(
                    predicted_key, true_key, ciphertext, true_plaintext
                )
                all_metrics.append(metrics)

    # Aggregate metrics
    agg = aggregate_metrics(all_metrics)

    return agg, all_metrics


def main():
    """
    Main evaluation function.
    """
    print("="*80)
    print("EVALUATING SUBSTITUTION CIPHER KEY PREDICTION MODEL")
    print("="*80)

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = CipherDataset(DATA_DIR / "test.json", max_src_len=MAX_SRC_LEN)

    # Limit to NUM_TEST_SAMPLES if specified
    if NUM_TEST_SAMPLES is not None:
        print(f"Evaluating on {NUM_TEST_SAMPLES} samples (for speed)")
        # Create subset
        test_dataset.data = test_dataset.data[:NUM_TEST_SAMPLES]

    vocab_size = test_dataset.vocab_size
    print(f"Test samples: {len(test_dataset)}")
    print(f"Vocabulary size: {vocab_size}")

    # Create data loader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Create model
    print("\nLoading model...")
    model = CryptoTransformer(
        vocab_size=vocab_size,
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
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Training val loss: {checkpoint['val_loss']:.4f}")

    # Evaluate
    agg, all_metrics = evaluate_model(model, test_loader, test_dataset, device)

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Key Accuracy:              {agg['key_accuracy']:.2f}%")
    print(f"Perfect Key Recovery Rate: {agg['perfect_key_recovery_rate']:.2f}%")
    print(f"Character Accuracy:        {agg['character_accuracy']:.2f}%")
    print(f"Word Accuracy:             {agg['word_accuracy']:.2f}%")

    # Show some examples
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS (first 5)")
    print("="*80)

    for i in range(min(5, len(all_metrics))):
        metrics = all_metrics[i]
        print(f"\nExample {i+1}:")
        print(f"  Key accuracy:       {metrics['key_accuracy']:.2f}%")
        print(f"  Perfect key:        {'✓' if metrics['perfect_key_recovery'] == 1.0 else '✗'}")
        print(f"  Character accuracy: {metrics['character_accuracy']:.2f}%")
        if len(metrics['predicted_plaintext']) > 0:
            print(f"  Decrypted text:     {metrics['predicted_plaintext'][:80]}...")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
