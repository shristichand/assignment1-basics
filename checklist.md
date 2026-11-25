CS336 A1 — FINISH CHECKLIST

TOKENIZER
[Done ] run_train_bpe
[Done ] BPE tests
[Done ] Tokenizer __init__
[Done ] encode
[Done ] decode
[Done ] special tokens
[Done ] tokenizer tests

TRANSFORMER
[ ] Linear
[ ] Embedding
[ ] RMSNorm
[ ] SiLU
[ ] Softmax
[ ] SwiGLU
[ ] SDPA
[ ] Multi-head attention
[ ] RoPE
[ ] MHA + RoPE
[ ] Transformer block
[ ] Transformer LM
[ ] model tests

TRAINING
[ ] get_batch
[ ] cross_entropy
[ ] gradient clipping
[ ] AdamW
[ ] LR schedule
[ ] checkpoint save
[ ] checkpoint load
[ ] training utility tests

TRAINING SCRIPT
[ ] load TinyStories
[ ] tokenize data
[ ] create model
[ ] create optimizer
[ ] training loop
[ ] validation
[ ] logging
[ ] checkpointing
[ ] TinyStories experiment
[ ] OpenWebText experiment

FINAL
[ ] uv run pytest
[ ] fix remaining failures
[ ] bash make_submission.sh