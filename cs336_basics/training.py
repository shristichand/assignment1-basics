"""
CS336 Assignment 1 — Training utilities.

Implement:
  - get_batch: sample a random batch from a token array
  - cross_entropy: numerically stable average cross-entropy
  - gradient_clipping: clip global L2 norm of gradients
  - AdamW: Adam with decoupled weight decay
  - cosine LR schedule: linear warmup + cosine decay
  - save / load checkpoint
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Batch sampling
# ---------------------------------------------------------------------------
def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample `batch_size` random windows of length `context_length` from `dataset`.

    Returns:
        inputs  (batch_size, context_length): token IDs
        targets (batch_size, context_length): next-token labels (inputs shifted right by 1)
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Cross-entropy loss
# ---------------------------------------------------------------------------
def cross_entropy(
    logits: torch.Tensor,   # (batch_size, vocab_size) — unnormalized
    targets: torch.Tensor,  # (batch_size,) — integer class indices
) -> torch.Tensor:
    """
    Compute the mean cross-entropy loss across the batch.
    Use the log-sum-exp trick for numerical stability.
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------
def clip_gradients(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip the combined L2 norm of all parameter gradients to `max_l2_norm`.
    Modifies .grad tensors in-place.
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------------------
# AdamW optimizer
# ---------------------------------------------------------------------------
class AdamW(torch.optim.Optimizer):
    """
    Adam optimizer with decoupled weight decay (Loshchilov & Hutter, 2019).

    Update rule (per parameter):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g²
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        θ = θ - lr * m_hat / (sqrt(v_hat) + eps)
        θ = θ - lr * weight_decay * θ    ← decoupled, applied AFTER Adam step
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # TODO: implement the update rule above
                raise NotImplementedError

        return loss


# ---------------------------------------------------------------------------
# Cosine LR schedule with linear warmup
# ---------------------------------------------------------------------------
def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Return the learning rate at iteration `it`:
      - Linear warmup from 0 to max_lr over `warmup_iters` steps
      - Cosine decay from max_lr to min_lr over `cosine_cycle_iters` steps
      - Constant min_lr after `cosine_cycle_iters`
    """
    # TODO
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Serialize model state, optimizer state, and iteration to `out`."""
    # TODO: use torch.save with a dict
    raise NotImplementedError


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load checkpoint from `src` into model and optimizer. Return iteration number."""
    # TODO: use torch.load, then load_state_dict on model and optimizer
    raise NotImplementedError
