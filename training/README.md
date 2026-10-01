# Training — Thursday AI

This folder contains everything needed to fine-tune `Qwen2.5-3B-Instruct` into **Thursday AI v0.1** on Kaggle T4×2.

## Files

- **`thursday_ai_finetune.ipynb`** — the Kaggle notebook (30 cells). Upload this to Kaggle, set accelerator to GPU T4×2, add `HF_TOKEN` secret, run all.
- **`requirements.txt`** — pinned dependency versions for reproducibility (the notebook installs these in-cell, this file is for reference).

## What the notebook does

1. Clones the Thursday AI repo from GitHub
2. Loads `Qwen/Qwen2.5-3B-Instruct` in 4-bit NF4 via Unsloth
3. Attaches LoRA adapters (r=32, alpha=64, all-linear targets)
4. Loads the dataset (template-generated + hand-written examples) and formats with Qwen2 tool-call chat template
5. Masks loss on everything except assistant turns (`train_on_responses_only`)
6. Trains 3 epochs at effective batch 16 (2 GPUs × batch 4 × grad_accum 2)
7. Saves LoRA adapter, merges into base
8. Converts to GGUF `Q4_K_M` via llama.cpp
9. Pushes merged model + GGUF to HuggingFace Hub
10. Smoke-tests the model on a "set alarm for 7am" prompt

## Expected wall-clock

| Dataset size | Steps | Time on T4×2 |
|---|---|---|
| 1k examples (smoke) | ~190 steps | ~5 min |
| 15k examples (Phase 1) | ~2,800 steps | ~75 min |
| 90k examples (Phase 2) | ~17,000 steps | ~6-8 hours |

Kaggle's session limit is 12 hours, so the full 90k run fits with margin.

## Secrets

Add to Kaggle Secrets:
- `HF_TOKEN` — HuggingFace write token (needed for upload)

The notebook uses `kaggle_secrets.UserSecretsClient` to load this; no other secrets needed.

## After training

The notebook uploads:
- `skyro777/thursday-ai-v0.1-merged` — merged HF model (for further fine-tuning)
- `skyro777/thursday-ai-v0.1-gguf` — the Q4_K_M GGUF (for inference)

On your potato PC:
```bash
ollama pull hf.co/skyro777/thursday-ai-v0.1-gguf:Q4_K_M
ollama run hf.co/skyro777/thursday-ai-v0.1-gguf:Q4_K_M
```

## Troubleshooting

- **OOM during training**: reduce `per_device_batch` from 4 to 2 in `CONFIG`. Effective batch will drop to 8.
- **Kaggle session times out**: enable checkpoint resume — the notebook saves every 500 steps to `/kaggle/working/thursday-ai-lora/checkpoint-*/`. Re-run the notebook with `trainer.train(resume_from_checkpoint=True)` to continue.
- **GGUF conversion fails**: ensure `llama.cpp` cloned successfully. The build step `make -C /kaggle/working/llama.cpp llama-quantize` requires ~2 GB disk; clean `/kaggle/working` if low.
- **`train_on_responses_only` import error**: your unsloth version may be older. Update with `pip install --upgrade "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`.
- **Notebook cannot find dataset**: the notebook clones the repo to `/kaggle/working/Thursday_AI`. If you forked, update `CONFIG['github_repo']` and `CONFIG['github_branch']` at the top of the notebook.
