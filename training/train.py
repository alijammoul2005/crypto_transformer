"""
Training script for substitution cipher Transformer.

Implements:
- Teacher forcing training
- Warmup learning rate schedule
- Early stopping
- Model checkpointing
- Progress logging
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import CipherDataset
from model.transformer import CryptoTransformer
from training.scheduler import WarmupScheduler

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Hyperparameters
BATCH_SIZE = 32
MAX_EPOCHS = 20
WARMUP_STEPS = 4000
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 4
D_FF = 512
DROPOUT = 0.1
MAX_LEN = 202
EARLY_STOPPING_PATIENCE = 3

# Paths
DATA_DIR = Path("/kaggle/working/data")
CHECKPOINT_DIR = Path("/kaggle/working/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def create_masks(src, tgt, pad_idx):
    """
    Create attention masks for source and target sequences.

    Args:
        src: Source tensor (batch, src_seq_len)
        tgt: Target tensor (batch, tgt_seq_len)
        pad_idx: Index of PAD token

    Returns:
        src_mask: Padding mask for source (batch, 1, 1, src_seq_len)
        tgt_mask: Combined causal and padding mask for target (batch, 1, tgt_seq_len, tgt_seq_len)
    """
    batch_size = src.size(0)
    src_seq_len = src.size(1)
    tgt_seq_len = tgt.size(1)

    # Source padding mask: mask out PAD tokens
    # Shape: (batch, 1, 1, src_seq_len)
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    # Target padding mask: mask out PAD tokens
    # Shape: (batch, 1, tgt_seq_len)
    tgt_padding_mask = (tgt != pad_idx).unsqueeze(1)

    # Target causal mask: prevent attending to future positions
    # Shape: (1, 1, tgt_seq_len, tgt_seq_len)
    tgt_causal_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len, device=tgt.device)).bool()
    tgt_causal_mask = tgt_causal_mask.unsqueeze(0).unsqueeze(0)

    # Combine causal and padding masks
    # Shape: (batch, 1, tgt_seq_len, tgt_seq_len)
    tgt_mask = tgt_causal_mask & tgt_padding_mask.unsqueeze(3)

    return src_mask, tgt_mask


def calculate_accuracy(logits, targets, pad_idx):
    """
    Calculate character-level accuracy, ignoring PAD tokens.

    Args:
        logits: Model output (batch, seq_len, vocab_size)
        targets: Ground truth (batch, seq_len)
        pad_idx: Index of PAD token

    Returns:
        Accuracy as percentage
    """
    predictions = torch.argmax(logits, dim=-1)

    # Create mask for non-PAD tokens
    mask = (targets != pad_idx)

    # Count correct predictions
    correct = ((predictions == targets) & mask).sum().item()
    total = mask.sum().item()

    if total == 0:
        return 0.0

    return 100.0 * correct / total


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, pad_idx, epoch):
    """
    Train for one epoch.

    Args:
        model: Transformer model
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Device (cuda or cpu)
        pad_idx: PAD token index
        epoch: Current epoch number

    Returns:
        Average loss for epoch
        Average accuracy for epoch
    """
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (src, tgt) in enumerate(progress_bar):
        src = src.to(device)
        tgt = tgt.to(device)

        # Split target into input and output
        # Input: <SOS> + plaintext (remove last token)
        # Output: plaintext + <EOS> (remove first token)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Create masks
        src_mask, tgt_mask = create_masks(src, tgt_input, pad_idx)

        # Forward pass
        logits = model(src, tgt_input, src_mask, tgt_mask)

        # Calculate loss
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Calculate accuracy
        accuracy = calculate_accuracy(logits, tgt_output, pad_idx)

        # Update statistics
        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.2f}%',
            'lr': f'{scheduler.get_lr():.6f}'
        })

        # Log every 100 steps
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / num_batches
            avg_acc = total_accuracy / num_batches
            print(f"  Step {batch_idx + 1}: loss={avg_loss:.4f}, acc={avg_acc:.2f}%, lr={scheduler.get_lr():.6f}")

    return total_loss / num_batches, total_accuracy / num_batches


def validate(model, dataloader, criterion, device, pad_idx):
    """
    Validate model on validation set.

    Args:
        model: Transformer model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device (cuda or cpu)
        pad_idx: PAD token index

    Returns:
        Average loss
        Average accuracy
    """
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Validating"):
            src = src.to(device)
            tgt = tgt.to(device)

            # Split target
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Create masks
            src_mask, tgt_mask = create_masks(src, tgt_input, pad_idx)

            # Forward pass
            logits = model(src, tgt_input, src_mask, tgt_mask)

            # Calculate loss and accuracy
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            accuracy = calculate_accuracy(logits, tgt_output, pad_idx)

            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1

    return total_loss / num_batches, total_accuracy / num_batches


def plot_training_curves(train_losses, val_losses, save_path):
    """
    Plot and save training curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")


def main():
    """
    Main training function.
    """
    print("="*80)
    print("TRANSFORMER TRAINING FOR SUBSTITUTION CIPHER CRYPTANALYSIS")
    print("="*80)

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    print()

    # Load datasets
    print("Loading datasets...")
    train_dataset = CipherDataset(DATA_DIR / "train.json", max_len=MAX_LEN)
    val_dataset = CipherDataset(DATA_DIR / "val.json", max_len=MAX_LEN)

    vocab_size = train_dataset.vocab_size
    pad_idx = train_dataset.PAD_IDX

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"PAD_IDX: {pad_idx}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create model
    print("\nInitializing model...")
    model = CryptoTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT
    ).to(device)

    print(f"Total parameters: {model.count_parameters():,}")

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, D_MODEL, WARMUP_STEPS)

    # Loss function (ignore PAD tokens)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{MAX_EPOCHS}")
        print(f"{'='*80}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, pad_idx, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, pad_idx)

        # Save losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
            }, checkpoint_path)
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")

        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")

            # Early stopping
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Plot training curves
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    plot_training_curves(train_losses, val_losses, CHECKPOINT_DIR / "training_curves.png")

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {CHECKPOINT_DIR / 'best_model.pt'}")


if __name__ == '__main__':
    main()
