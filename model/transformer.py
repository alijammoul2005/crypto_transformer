"""
Full Transformer model for substitution cipher cryptanalysis.

Encoder-decoder architecture built from scratch following
"Attention is All You Need" (Vaswani et al., 2017).
"""

import torch
import torch.nn as nn
import math
from .attention import MultiHeadAttention


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """
    Single Transformer encoder layer.

    Components:
    1. Multi-head self-attention
    2. Feed-forward network
    3. Layer normalization and residual connections

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward network dimension
        dropout: Dropout probability
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass of encoder layer.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional padding mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DecoderLayer(nn.Module):
    """
    Single Transformer decoder layer.

    Components:
    1. Masked multi-head self-attention
    2. Multi-head cross-attention (over encoder output)
    3. Feed-forward network
    4. Layer normalization and residual connections

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward network dimension
        dropout: Dropout probability
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Masked self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Cross-attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        Forward pass of decoder layer.

        Args:
            x: Decoder input (batch, tgt_seq_len, d_model)
            encoder_output: Encoder output (batch, src_seq_len, d_model)
            tgt_mask: Causal mask for decoder self-attention
            src_mask: Padding mask for encoder output

        Returns:
            Output tensor (batch, tgt_seq_len, d_model)
        """
        # Masked self-attention with residual connection and layer norm
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # Cross-attention with residual connection and layer norm
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.ff(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class CryptoTransformer(nn.Module):
    """
    Full Transformer model for substitution cipher KEY prediction.

    NEW APPROACH: Predicts the 26-letter substitution key instead of plaintext.

    Encoder-decoder architecture with:
    - Character-level embeddings
    - Sinusoidal positional encoding
    - Multi-head attention
    - Feed-forward networks

    Input: ciphertext (200 chars)
    Output: substitution key (26 letters)

    Args:
        vocab_size: Size of vocabulary (30 for a-z + space + special tokens)
        d_model: Model dimension (256)
        num_heads: Number of attention heads (4)
        num_layers: Number of encoder/decoder layers (4)
        d_ff: Feed-forward dimension (1024)
        max_len: Maximum sequence length (201 for src, 28 for tgt)
        dropout: Dropout probability (0.1)
    """

    def __init__(self, vocab_size=30, d_model=256, num_heads=4, num_layers=4,
                 d_ff=1024, max_len=201, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embeddings
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        """
        Encode source sequence.

        Args:
            src: Source tokens (batch, src_seq_len)
            src_mask: Source padding mask

        Returns:
            Encoder output (batch, src_seq_len, d_model)
        """
        # Embedding and positional encoding
        x = self.encoder_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(self, tgt, encoder_output, tgt_mask=None, src_mask=None):
        """
        Decode target sequence.

        Args:
            tgt: Target tokens (batch, tgt_seq_len)
            encoder_output: Encoder output (batch, src_seq_len, d_model)
            tgt_mask: Causal mask for target
            src_mask: Source padding mask

        Returns:
            Decoder output (batch, tgt_seq_len, d_model)
        """
        # Embedding and positional encoding
        x = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)

        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass of full model.

        Args:
            src: Source tokens (batch, src_seq_len)
            tgt: Target tokens (batch, tgt_seq_len)
            src_mask: Source padding mask
            tgt_mask: Causal mask for target

        Returns:
            Logits over vocabulary (batch, tgt_seq_len, vocab_size)
        """
        # Encode source
        encoder_output = self.encode(src, src_mask)

        # Decode target
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)

        # Project to vocabulary
        logits = self.output_projection(decoder_output)

        return logits

    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """
    Test function for transformer model.
    """
    batch_size = 2
    src_seq_len = 50
    tgt_seq_len = 50
    vocab_size = 30

    # Create model
    model = CryptoTransformer(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        max_len=202,
        dropout=0.1
    )

    print(f"Total parameters: {model.count_parameters():,}")

    # Random input
    src = torch.randint(0, vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))

    # Forward pass
    logits = model(src, tgt)
    print(f"Input source shape: {src.shape}")
    print(f"Input target shape: {tgt.shape}")
    print(f"Output logits shape: {logits.shape}")


if __name__ == '__main__':
    test_model()
