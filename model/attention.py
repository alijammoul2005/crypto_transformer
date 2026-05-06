"""
Multi-head attention mechanism for Transformer.

Implements scaled dot-product attention with multiple heads.
"""

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Splits input into multiple heads, computes attention for each head,
    then concatenates and projects the results.

    Args:
        d_model: Model dimension (embedding size)
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.

        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

        Args:
            Q: Query tensor (batch, num_heads, seq_len, d_k)
            K: Key tensor (batch, num_heads, seq_len, d_k)
            V: Value tensor (batch, num_heads, seq_len, d_k)
            mask: Optional mask tensor (batch, 1, seq_len, seq_len) or (batch, 1, 1, seq_len)

        Returns:
            Attention output (batch, num_heads, seq_len, d_k)
            Attention weights (batch, num_heads, seq_len, seq_len)
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (set masked positions to large negative value)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)

        return output, attn_weights

    def split_heads(self, x):
        """
        Split input into multiple heads.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Tensor with shape (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        Combine multiple heads back into single tensor.

        Args:
            x: Input tensor (batch, num_heads, seq_len, d_k)

        Returns:
            Tensor with shape (batch, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor (batch, seq_len_q, d_model)
            key: Key tensor (batch, seq_len_k, d_model)
            value: Value tensor (batch, seq_len_v, d_model)
            mask: Optional mask tensor

        Returns:
            Output tensor (batch, seq_len_q, d_model)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)  # (batch, seq_len_q, d_model)
        K = self.W_k(key)    # (batch, seq_len_k, d_model)
        V = self.W_v(value)  # (batch, seq_len_v, d_model)

        # Split into multiple heads
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len_q, d_k)
        K = self.split_heads(K)  # (batch, num_heads, seq_len_k, d_k)
        V = self.split_heads(V)  # (batch, num_heads, seq_len_v, d_k)

        # Compute attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads
        output = self.combine_heads(attn_output)  # (batch, seq_len_q, d_model)

        # Final linear projection
        output = self.W_o(output)

        return output


def test_attention():
    """
    Test function for multi-head attention.
    """
    batch_size = 2
    seq_len = 10
    d_model = 256
    num_heads = 8

    # Create attention module
    attn = MultiHeadAttention(d_model, num_heads)

    # Random input
    x = torch.randn(batch_size, seq_len, d_model)

    # Test self-attention
    output = attn(x, x, x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test with mask
    mask = torch.tril(torch.ones(1, 1, seq_len, seq_len))
    output_masked = attn(x, x, x, mask)
    print(f"Output with mask shape: {output_masked.shape}")


if __name__ == '__main__':
    test_attention()
