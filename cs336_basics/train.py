"""
CS336 Assignment 1 — Main training script.

Fill in the TODOs to train your TransformerLM on TinyStories or OpenWebText.
Run with:
    uv run cs336_basics/train.py

Optional: integrate W&B for logging (already listed as a dependency).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

# TODO: import your model and training utilities once you've implemented them
# from cs336_basics.model import TransformerLM
# from cs336_basics.training import (
#     AdamW, get_batch, cross_entropy,
#     clip_gradients, get_lr_cosine_schedule,
#     save_checkpoint, load_checkpoint,
# )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT-style LM.")

    # Data
    parser.add_argument("--train_path", type=str, required=True, help="Path to tokenized train data (.npy)")
    parser.add_argument("--val_path", type=str, default=None, help="Path to tokenized validation data (.npy)")

    # Model
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1344)  # ~(8/3) * d_model, rounded
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--resume_from", type=str, default=None)

    # Misc
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--use_wandb", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # -------------------------------------------------------------------
    # 1. Load tokenized data
    # -------------------------------------------------------------------
    # TODO: load your pre-tokenized numpy array from args.train_path
    # Hint: use np.load(..., mmap_mode='r') for large files
    train_data = None  # TODO

    val_data = None
    if args.val_path:
        val_data = None  # TODO

    # -------------------------------------------------------------------
    # 2. Build model
    # -------------------------------------------------------------------
    # TODO: instantiate TransformerLM with args.*
    model = None  # TODO
    # model = model.to(args.device)
    # print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -------------------------------------------------------------------
    # 3. Build optimizer
    # -------------------------------------------------------------------
    # TODO: instantiate your AdamW with args.lr, args.weight_decay, etc.
    optimizer = None  # TODO

    # -------------------------------------------------------------------
    # 4. Optionally resume from checkpoint
    # -------------------------------------------------------------------
    start_iter = 0
    if args.resume_from:
        pass  # TODO: call load_checkpoint

    # -------------------------------------------------------------------
    # 5. W&B (optional)
    # -------------------------------------------------------------------
    if args.use_wandb:
        import wandb
        wandb.init(project="cs336-assignment1", config=vars(args))

    # -------------------------------------------------------------------
    # 6. Training loop
    # -------------------------------------------------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for it in range(start_iter, args.max_iters):

        # --- Learning rate schedule ---
        lr = None  # TODO: call get_lr_cosine_schedule
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr

        # --- Sample batch ---
        # inputs, targets = get_batch(train_data, args.batch_size, args.context_length, args.device)

        # --- Forward pass ---
        # logits = model(inputs)
        # Reshape logits and targets for cross_entropy
        # loss = cross_entropy(logits.view(-1, args.vocab_size), targets.view(-1))

        # --- Backward pass ---
        # optimizer.zero_grad()
        # loss.backward()

        # --- Gradient clipping ---
        # clip_gradients(model.parameters(), args.max_grad_norm)

        # --- Optimizer step ---
        # optimizer.step()

        # --- Logging ---
        if it % args.eval_every == 0:
            pass  # TODO: evaluate on val_data and log

        # --- Checkpoint ---
        if it % args.save_every == 0 and it > 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_{it:06d}.pt")
            pass  # TODO: call save_checkpoint

    print("Training complete.")


if __name__ == "__main__":
    main()
