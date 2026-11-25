"""
CS336 Assignment 1 — Transformer model components.

Implement each component as a torch.nn.Module.
The tests call your implementations via tests/adapters.py.

Reading order (suggested):
  1. Linear, Embedding
  2. RMSNorm, SiLU
  3. SwiGLU
  4. Softmax, ScaledDotProductAttention
  5. MultiHeadSelfAttention (no RoPE)
  6. RoPE, MultiHeadSelfAttentionWithRoPE
  7. TransformerBlock
  8. TransformerLM
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Linear (no bias)
# ---------------------------------------------------------------------------
class Linear(nn.Module):
    """A linear transformation y = xW^T (no bias)."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()

        # Trainable weight: shape [d_out, d_in]
        self.weight = nn.Parameter(torch.empty(d_out, d_in))

        # Initialize according to the assignment specification
        std = (2 / (d_in + d_out)) ** 0.5
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x):
        return x @ self.weight.T

# ---------------------------------------------------------------------------
# 2. Embedding
# ---------------------------------------------------------------------------
class Embedding(nn.Module):
    """Lookup table mapping integer token IDs → dense vectors."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))

    def forward(self, token_ids):
        return self.weight[token_ids]


# ---------------------------------------------------------------------------
# 3. RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # TODO: define self.weight (learnable scale)
        self.weight = nn.Parameter(torch.empty(d_model))

    def forward(self, x):
        # TODO
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# ---------------------------------------------------------------------------
# 4. SiLU activation
# ---------------------------------------------------------------------------
def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Sigmoid Linear Unit): x * sigmoid(x)."""
    # TODO
    return x * torch.sigmoid(x)


# ---------------------------------------------------------------------------
# 5. SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """
    SwiGLU FFN: output = W2(SiLU(W1 x) * W3 x)

    W1, W3: (d_ff, d_model)
    W2:     (d_model, d_ff)
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))
    
# ---------------------------------------------------------------------------
# 6. Softmax (numerically stable)
# ---------------------------------------------------------------------------
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically stable softmax along `dim`."""
    # TODO: subtract max before computing exp
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

# ---------------------------------------------------------------------------
# 7. Scaled Dot-Product Attention
# ---------------------------------------------------------------------------
class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    Optional boolean mask: True = keep, False = mask out (set to -inf).
    """

    def forward(self, Q, K, V, mask=None):
        # TODO
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 8. Multi-Head Self-Attention (no RoPE)
# ---------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention WITHOUT positional encoding.
    Uses a single batched matrix multiply for all heads' Q, K, V projections.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # TODO: define q_proj, k_proj, v_proj, output_proj

    def forward(self, x):
        # TODO: project, split heads, SDPA, merge heads, output project
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 9. Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------
class RoPE(nn.Module):
    """
    Rotary Position Embeddings.
    Pre-compute sin/cos tables up to max_seq_len.
    """

    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        # TODO: compute and register sin/cos buffers

    def forward(self, x, token_positions):
        # TODO: apply rotation
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 10. Multi-Head Self-Attention WITH RoPE
# ---------------------------------------------------------------------------
class MultiHeadSelfAttentionWithRoPE(nn.Module):
    """Multi-head self-attention with RoPE applied to Q and K."""

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float):
        super().__init__()
        # TODO

    def forward(self, x, token_positions=None):
        # TODO
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 11. Transformer Block (pre-norm)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block:
      x = x + MHA(RMSNorm(x))
      x = x + FFN(RMSNorm(x))
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        # TODO: define ln1, attn, ln2, ffn

    def forward(self, x, token_positions=None):
        # TODO
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 12. Full Transformer Language Model
# ---------------------------------------------------------------------------
class TransformerLM(nn.Module):
    """
    GPT-style language model:
      token_embeddings → N × TransformerBlock → RMSNorm → lm_head (linear)
    Weight tying: token_embeddings.weight == lm_head.weight
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        # TODO: define token_embeddings, layers (nn.ModuleList), ln_final, lm_head

    def forward(self, token_ids):
        # TODO
        raise NotImplementedError
