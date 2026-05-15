"""
Training script for substitution cipher KEY prediction Transformer.

NEW APPROACH: Predict the 26-letter substitution key, then apply deterministically.

Implements:
- Teacher forcing training
- Warmup learning rate schedule
- Early stopping
- Model checkpointing
- Progress logging
- Fresh key generation per epoch for maximum diversity
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
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

# Hyperparameters - Optimized for Google Colab (T4 GPU ~15GB VRAM)
BATCH_SIZE = 128  # Optimized for Colab T4 GPU
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 256
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS  # 256
MAX_EPOCHS = 20
WARMUP_STEPS = 2000  # Faster warmup for simpler task
D_MODEL = 256  # Efficient size for key prediction
NUM_HEADS = 4
NUM_LAYERS = 4
D_FF = 1024
DROPOUT = 0.1
MAX_SRC_LEN = 201  # Ciphertext: 200 chars + EOS
MAX_TGT_LEN = 28   # Key: SOS + 26 letters + EOS
EARLY_STOPPING_PATIENCE = 5
USE_MIXED_PRECISION = True  # FP16 training to save memory (~40% reduction)
NUM_WORKERS = 2  # Colab has 2 CPU cores typically

# Paths - use relative paths for local training
SCRIPT_DIR = Path(__file__).parent.parent  # crypto_transformer directory
DATA_DIR = SCRIPT_DIR / "data" / "generated"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
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


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, pad_idx, epoch, scaler):
    """
    Train for one epoch with gradient accumulation and mixed precision.

    Args:
        model: Transformer model
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Device (cuda or cpu)
        pad_idx: PAD token index
        epoch: Current epoch number
        scaler: GradScaler for mixed precision

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

        # Forward pass with mixed precision
        with autocast(enabled=USE_MIXED_PRECISION and torch.cuda.is_available()):
            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            # Scale loss for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Update weights every GRADIENT_ACCUMULATION_STEPS
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # Calculate accuracy
        with torch.no_grad():
            accuracy = calculate_accuracy(logits, tgt_output, pad_idx)

        # Update statistics (use unscaled loss)
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        total_accuracy += accuracy
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}',
            'acc': f'{accuracy:.2f}%',
            'lr': f'{scheduler.get_lr():.6f}',
            'mem': f'{torch.cuda.memory_allocated()/1e9:.1f}GB' if torch.cuda.is_available() else 'N/A'
        })

        # Log every 100 steps
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / num_batches
            avg_acc = total_accuracy / num_batches
            print(f"  Step {batch_idx + 1}: loss={avg_loss:.4f}, acc={avg_acc:.2f}%, lr={scheduler.get_lr():.6f}", flush=True)

        # Print GPU memory stats after first training step
        if batch_idx == 0 and epoch == 1 and torch.cuda.is_available():
            print(f"  After first step - Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
            print(f"  After first step - Memory reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB", flush=True)
            print(f"  Peak memory:      {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)

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
    print(f"Training curves saved to {save_path}", flush=True)


def main():
    """
    Main training function.
    """
    print("="*80, flush=True)
    print("TRANSFORMER TRAINING FOR SUBSTITUTION CIPHER CRYPTANALYSIS", flush=True)
    print("="*80, flush=True)

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"Total VRAM: {gpu_props.total_memory / 1e9:.2f} GB", flush=True)
        print(f"CUDA Capability: {gpu_props.major}.{gpu_props.minor}", flush=True)
        print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
        print(f"Memory reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB", flush=True)
    print(f"\nBatch size: {BATCH_SIZE}", flush=True)
    print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}", flush=True)
    print(f"Effective batch size: {EFFECTIVE_BATCH_SIZE}", flush=True)
    print(f"Mixed precision (FP16): {USE_MIXED_PRECISION}", flush=True)
    print(flush=True)

    # Load datasets
    print("Loading datasets...", flush=True)
    train_dataset = CipherDataset(DATA_DIR / "train.json", max_src_len=MAX_SRC_LEN)
    val_dataset = CipherDataset(DATA_DIR / "val.json", max_src_len=MAX_SRC_LEN)

    vocab_size = train_dataset.vocab_size
    pad_idx = train_dataset.PAD_IDX

    print(f"Training samples: {len(train_dataset)}", flush=True)
    print(f"Validation samples: {len(val_dataset)}", flush=True)
    print(f"Vocabulary size: {vocab_size}", flush=True)
    print(f"PAD_IDX: {pad_idx}", flush=True)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Training batches: {len(train_loader)}", flush=True)
    print(f"Validation batches: {len(val_loader)}", flush=True)

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create model
    print("\nInitializing model...", flush=True)
    print(f"Task: Predict 26-letter substitution key from ciphertext", flush=True)
    model = CryptoTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len=max(MAX_SRC_LEN, MAX_TGT_LEN),
        dropout=DROPOUT
    )

    # Create optimizer (before loading checkpoint)
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, D_MODEL, WARMUP_STEPS)

    # Create GradScaler for mixed precision training
    scaler = GradScaler(enabled=USE_MIXED_PRECISION and torch.cuda.is_available())

    # Check for existing checkpoint to resume from
    resume_checkpoint = CHECKPOINT_DIR / "best_model.pt"
    start_epoch = 1
    best_val_loss = float('inf')
    if resume_checkpoint.exists():
        print(f"Found checkpoint: {resume_checkpoint}", flush=True)
        print("Loading checkpoint...", flush=True)
        checkpoint = torch.load(resume_checkpoint, map_location='cpu')

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint and USE_MIXED_PRECISION:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']

        print(f"Resumed from epoch {checkpoint['epoch']}, best val_loss: {best_val_loss:.4f}", flush=True)

    # Move model to device (single GPU optimized for RTX 4060)
    model = model.to(device)

    print(f"Total parameters: {model.count_parameters():,}", flush=True)

    # Print memory usage after model initialization
    if torch.cuda.is_available():
        print(f"Memory after model init: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB reserved", flush=True)

    # Loss function (ignore PAD tokens, NO label smoothing for exact key recovery)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.0)

    # Training loop
    print("\n" + "="*80, flush=True)
    print("STARTING TRAINING", flush=True)
    print("="*80, flush=True)

    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        print(f"\n{'='*80}", flush=True)
        print(f"EPOCH {epoch}/{MAX_EPOCHS}", flush=True)
        print(f"{'='*80}", flush=True)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, pad_idx, epoch, scaler
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, pad_idx)

        # Save losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:", flush=True)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%", flush=True)
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%", flush=True)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
            }, checkpoint_path)
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})", flush=True)
            if torch.cuda.is_available():
                print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB ({100*torch.cuda.memory_allocated()/torch.cuda.get_device_properties(0).total_memory:.1f}%)", flush=True)

        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)", flush=True)

            # Early stopping
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs", flush=True)
                break

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Plot training curves
    print("\n" + "="*80, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print("="*80, flush=True)
    plot_training_curves(train_losses, val_losses, CHECKPOINT_DIR / "training_curves.png")

    print(f"\nBest validation loss: {best_val_loss:.4f}", flush=True)
    print(f"Model saved to: {CHECKPOINT_DIR / 'best_model.pt'}", flush=True)


if __name__ == '__main__':
    main()
