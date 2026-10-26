# Checkpoints (not tracked by git)

    python scripts/download_gpt2.py gpt2          # gpt2.pt, 124M, for development and unit tests
    python scripts/download_gpt2.py gpt2-medium   # gpt2-medium.pt, 355M, for core experiments

Core experiments use base OpenAI gpt2-medium, unchanged. Setup downloads it to this folder.
The harness picks the first that exists: `$CPEN455_CKPT`, then
`gpt2-medium.pt`, `gpt2.pt` in this folder.
