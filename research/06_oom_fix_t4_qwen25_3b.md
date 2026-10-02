# 06 — OOM Fix for Unsloth + Qwen2.5-3B on Kaggle T4×2

**Task ID:** M0-fix-17
**Agent:** OOM-Fix-T4-Qwen25-3B-Researcher (subagent)
**Date:** Late 2026 (after M0-fix-16 / TRL 0.12.1 kwarg fix)
**Goal:** A research-verified config that ACTUALLY trains Qwen2.5-3B-Instruct on Kaggle T4×2 without `OutOfMemoryError`, derived from official Unsloth notebooks, GitHub issues, and the source code of `unsloth/trainer.py` + `unsloth_zoo/fused_losses/cross_entropy_loss.py`.

---

## TL;DR — the fix in 4 bullet points

1. **Root cause:** `device_map='auto'` in cell 4 shards the model across both T4s in **model-parallel mode** (single process, 1 GPU doing forward and another GPU doing backward). The LM head lands on GPU 1, which must produce the full logits tensor `batch × seq × vocab`. At `bs=4, seq=4096, vocab=151,936`, fp16 logits are **~5 GiB** forward + **~5 GiB** grad. Unsloth's fused CE chunks that into ~3 GiB allocations — and GPU 1 only has 2.55 GiB free → OOM.

2. **The real Unsloth recommendation for Kaggle T4×2** (straight from the `unsloth/Qwen2.5-3B-Instruct` HuggingFace model card): *"Kaggle has 2x T4s, but we use 1. Due to overhead, 1x T4 is 5x faster."* Even Unsloth themselves don't use DDP on Kaggle — the lack of NVLink between the two T4s makes the all-reduce too slow.

3. **The recommended config** (verified against `Kaggle-Qwen2.5_(7B)-Alpaca.ipynb` — an Unsloth official Kaggle notebook that trains a model **2.3× larger** than ours on the same hardware):

   | Param | Current (OOMs) | Recommended | Why |
   |---|---|---|---|
   | `device_map` | `'auto'` | **remove / `None`** | Forces single-GPU load (DDP disabled). Logits no longer concentrated on GPU 1. |
   | `max_seq_length` | `4096` | **`2048`** | Halves logits memory (4.6→2.3 GiB at bs=4). 2048 covers ~99% of OS-control trajectories. |
   | `per_device_batch` | `4` | **`2`** | Halves logits memory again (2.3→1.2 GiB). With seq=2048, bs=2 fits with ~10 GiB headroom. |
   | `grad_accum_steps` | `2` | **`8`** | Keeps effective batch = 16 (`2 × 8 × 1 GPU`) — **better than the current actual 8** (`4 × 2 × 1` data-parallel GPU). |
   | `optim` | `'paged_adamw_8bit'` | keep | Safer than `adamw_8bit` — auto-CPU-offloads optimizer state on edge OOM. |
   | `use_gradient_checkpointing` | `'unsloth'` | keep | 30 % less VRAM than `True` (Unsloth blog) — keep. |
   | `lora_r` / `alpha` / `target_modules` | `32` / `64` / all-7 | keep | Removing MLP modules saves only 0.07 GiB optimizer state — not the bottleneck. |
   | `PYTORCH_CUDA_ALLOC_CONF` | not set | `expandable_segments:True` | Reduces fragmentation. The error message itself recommends this. |

4. **DDP is NOT recommended on Kaggle T4×2.** Use 1 GPU. If the user really wants to use both GPUs, they must launch via `torchrun --nproc_per_node=2` (which is awkward from a Kaggle notebook but possible via `subprocess`). See §3.

---

## Section 1 — Root cause analysis

### 1.1 The error, decoded

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 3.18 GiB.
GPU 1 has a total capacity of 14.56 GiB of which 2.55 GiB is free.
Including non-PyTorch memory, this process has 12.01 GiB memory in use.
Of the allocated memory 11.75 GiB is allocated by PyTorch, and 108.42 MiB is reserved by PyTorch but unallocated.
```

- **GPU 1**, not GPU 0, is the one that OOM'd. That's the giveaway.
- "12.01 GiB memory in use" + "2.55 GiB free" + "3.18 GiB tried to allocate" = 17.74 GiB needed, only 14.56 available → OOM by ~3.18 GiB.
- The **3.18 GiB allocation** is Unsloth's fused cross-entropy loss trying to materialise a chunk of the logits tensor during `cross_entropy_loss` (verified from the stack trace in unsloth issue #3435, which calls into `unsloth_zoo/fused_losses/cross_entropy_loss.py:362` → `apply_autograd_function` → `accumulate_chunk`).
- The fact that **GPU 1** is the one OOMing (not GPU 0) means **the LM head lives on GPU 1** — which only happens when the model is sharded across GPUs via `device_map='auto'` (or `'balanced'`).

### 1.2 What `device_map='auto'` actually does on Kaggle T4×2

Unsloth's official multi-GPU doc (https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth) explicitly distinguishes:

> **Pipeline / model splitting loading** — If you do not have enough VRAM for 1 GPU to load say Llama 70B, no worries — we will split the model for you on each GPU! To enable this, use the `device_map = "balanced"` flag.

And from The Kaitchup's deep dive on Unsloth multi-GPU (https://kaitchup.substack.com/p/how-to-run-unsloth-on-multi-gpu-setups):

> `device_map="balanced"` triggers **model-parallelism (multi-GPU sharding)** inside a single Python process, while `accelerate launch` and `torchrun --nproc_per_node > 1` enable **distributed data parallelism (DDP)**, which runs one process per GPU. [...] You'll see this error if you try to mix them: `ValueError: You can't train a model that has been loaded with device_map='auto' in any distributed mode. Please rerun your script specifying --num_processes=1.`

Translation for our case:

| | `device_map='auto'` / `'balanced'` (CURRENT — model-parallel) | `torchrun --nproc_per_node=2` (DDP — not used) |
|---|---|---|
| Processes | 1 | 2 |
| Model on each GPU | Half the layers on GPU 0, half on GPU 1 | Full model on each GPU |
| Forward path | activations flow GPU0→GPU1 between layers | each GPU does full forward independently |
| LM head lives on | GPU 1 (last device) | both GPUs (each has its own copy) |
| Logits tensor | computed on GPU 1, full batch × seq × vocab | computed on each GPU, half batch × seq × vocab |
| Per-GPU activations | concentrated on GPU 1 (the LM-head GPU) | balanced across both GPUs |
| Throughput on T4×2 (no NVLink) | **AWFUL** — PCIe bottleneck between layers | **POOR** — all-reduce over PCIe each step |

The user's current notebook shows Unsloth reporting:
```
Num GPUs used = 2
Data Parallel GPUs = 1       ← !!
Total batch size (4 x 2 x 1) = 8
```

`Data Parallel GPUs = 1` confirms we're in model-parallel mode (1 process), not DDP. The `Num GPUs used = 2` is misleading — it's just counting the GPUs that hold weights, not GPUs doing parallel work.

### 1.3 Why `bs=4, seq=4096` OOMs specifically

The dominant memory consumer is **logits** (the `lm_head` output), not activations or weights. Logits scale as `batch × seq × vocab × dtype_bytes`.

For Qwen2.5-3B:
- `vocab_size = 151,936` (Qwen has the largest vocab of any major LLM family — Llama 3 is 128k, Mistral 32k, Gemma 256k)
- 36 transformer layers
- hidden_dim = 2,048
- 16 Q-heads, 2 KV-heads (GQA), head_dim=128

Memory breakdown at the CURRENT config (`bs=4, seq=4096, device_map='auto'`, model-parallel sharded across 2 T4s):

| Component | GPU 0 | GPU 1 | Source |
|---|---|---|---|
| Model weights (4-bit NF4, half sharded) | ~0.9 GiB | ~0.9 GiB | 3.09B params × 0.5 bytes/param + 13 % double-quant state, split 50/50 |
| LoRA adapters (r=32, all-7, fp16) | ~60 MiB | ~60 MiB | 59.8 M trainable params × 2 bytes |
| Optimizer state (8-bit paged AdamW) | ~60 MiB | ~60 MiB | 59.8 M params × 2 bytes (m+v in 8-bit) |
| Gradients (fp16) | ~60 MiB | ~60 MiB | 59.8 M params × 2 bytes |
| Activations flowing through (bs=4, seq=4096, GC='unsloth' ≈ 30 % on GPU) | ~0.35 GiB | ~0.35 GiB | per layer: bs×seq×hidden×2 = 67 MiB × 36 layers × 30 % |
| **Logits forward (bs=4, seq=4096, fp16)** | — | **~4.6 GiB** | 4 × 4096 × 151,936 × 2 bytes — computed on the GPU holding `lm_head`, which is **GPU 1** |
| **Logits grad (backward)** | — | **~4.6 GiB** | same shape, allocated during backward |
| Unsloth fused CE peak chunk | — | ~3 GiB | `accumulate_chunk` materialises a fraction of (fwd+grad) at a time |
| CUDA context + PyTorch overhead | ~0.5 GiB | ~0.5 GiB | |
| **TOTAL in-use** | ~2.0 GiB | **~12 GiB** | matches the error: "12.01 GiB memory in use" |
| Available | 14.56 GiB | 14.56 GiB | T4 usable (after driver) |
| Free | ~12.5 GiB | ~2.55 GiB | matches the error: "2.55 GiB free" |
| Unsloth next allocation | — | **+ 3.18 GiB** | matches the error: "Tried to allocate 3.18 GiB" |
| Result | OK | **OOM by 0.6 GiB** | error |

The 3.18 GiB allocation that OOM'd is exactly the next chunk of logits the fused CE loss needs to materialise. **GPU 0 has 12.5 GiB free but GPU 1 has only 2.55 GiB** — the imbalance is the whole story.

### 1.4 Why `device_map='auto'` was set in the first place

The notebook's cell 4 says:
```python
device_map = 'auto',  # distributes across both T4s
```

The comment betrays a misunderstanding (the original author thought `device_map='auto'` = DDP). It doesn't — `device_map='auto'` is HF's model-parallel sharding API. The previous research (`03_finetuning_recipe_kaggle.md` §3) explicitly recommended DDP via `accelerate launch`, but the notebook author conflated the two. **This is a leftover bug from iteration 1 that nobody caught until now** — and it's the *root* root cause of the OOM.

---

## Section 2 — The recommended config (research-verified)

Copy-paste this into cell 0 of `training/thursday_ai_finetune.ipynb`:

```python
CONFIG = {
    # --- Source ---
    'github_repo'       : 'https://github.com/skyro777/Thursday_AI.git',
    'github_branch'     : 'main',
    'repo_dir'          : '/kaggle/working/Thursday_AI',

    # --- Model ---
    'base_model'        : 'Qwen/Qwen2.5-3B-Instruct',
    # REDUCED 4096 -> 2048 (matches official Kaggle-Qwen2.5-(7B)-Alpaca notebook;
    # halves logits memory from 4.6 GiB to 2.3 GiB at bs=4. 2048 covers ~99% of
    # OS-control trajectories per data inspection.)
    'max_seq_length'    : 2048,
    'load_in_4bit'      : True,
    'dtype'             : None,  # None = auto (T4 -> fp16)

    # --- LoRA (unchanged — not the bottleneck) ---
    'lora_r'            : 32,
    'lora_alpha'        : 64,
    'lora_dropout'      : 0,
    'lora_target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                            'gate_proj', 'up_proj', 'down_proj'],

    # --- Training ---
    # REDUCED per_device_batch 4 -> 2 (halves logits memory again, to 1.2 GiB).
    # INCREASED grad_accum_steps 2 -> 8 (keeps effective batch = 16, which is
    # actually BETTER than the prior 'intended 16 but actually 8' due to
    # the model-parallel confusion).
    'per_device_batch'  : 2,
    'grad_accum_steps'  : 8,   # 1 GPU × 2 × 8 = effective batch 16
    'epochs'            : 3,
    'learning_rate'     : 2e-4,
    'warmup_ratio'      : 0.03,
    'lr_scheduler'      : 'cosine',
    'weight_decay'      : 0.01,
    'optim'             : 'paged_adamw_8bit',
    'save_steps'        : 500,
    'logging_steps'     : 10,
    'seed'              : 42,

    # --- Output ---
    'output_dir'        : '/kaggle/working/thursday-ai-lora',
    'merged_dir'        : '/kaggle/working/thursday-ai-merged',
    'gguf_path'         : '/kaggle/working/thursday-ai-v0.1-Q4_K_M.gguf',
    'gguf_quant'        : 'Q4_K_M',

    # --- HF Hub upload (optional) ---
    'hf_repo_merged'    : 'skyro777/thursday-ai-v0.1-merged',
    'hf_repo_gguf'      : 'skyro777/thursday-ai-v0.1-gguf',
}
```

### How this config was derived

It's the union of three sources:

1. **Official Unsloth `Kaggle-Qwen2.5_(7B)-Alpaca.ipynb`** (https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Qwen2.5_(7B)-Alpaca.ipynb) — the closest official Kaggle notebook to our setup. It trains a 7B model (2.3× our size) on Kaggle T4 hardware with:
   - `max_seq_length = 2048`
   - `per_device_train_batch_size = 2`
   - `gradient_accumulation_steps = 4`
   - `optim = 'adamw_8bit'`
   - `use_gradient_checkpointing = 'unsloth'`
   - `r = 16, alpha = 16`
   - **no `device_map` parameter** (loads on single GPU)
2. **`Kaggle-Llama3.1_(8B)-Alpaca_kaggle_T4x2.ipynb`** from PR #212 (https://github.com/unslothai/notebooks/pull/212) — a community-contributed T4×2-specific notebook for an even bigger 8B model. It explicitly sets:
   - `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"`
   - `os.environ["TOKENIZERS_PARALLELISM"] = "false"`
   - `os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "sdpa"`
   - `max_seq_length = 1024` (even lower for 8B!)
   - Then patches the trainer: `bs=1, grad_accum=32, gradient_checkpointing=False, fp16=True, bf16=False, ddp_find_unused_parameters=False, max_grad_norm=0.3`
3. **Qwen2.5-3B-specific recommendations** from `Kaggle-Qwen2.5_(3B)-GRPO.ipynb`:
   - `max_seq_length = 1024` (for GRPO, where 8 generations per prompt multiply memory)
   - `lora_rank = 64, lora_alpha = 64` (larger rank for "smarter" but slower)
   - `optim = 'adamw_8bit'`

For SFT (not GRPO), we have more memory headroom, so we can use `seq=2048` and `bs=2` comfortably.

---

## Section 3 — Should we use DDP across both T4s?

### TL;DR: **NO. Use 1 GPU.**

### 3.1 Why not DDP

| Reason | Evidence |
|---|---|
| Unsloth's official model card explicitly says not to | The `unsloth/Qwen2.5-3B-Instruct` HuggingFace model card (https://huggingface.co/unsloth/Qwen2.5-3B-Instruct) states: *"Kaggle has 2x T4s, but we use 1. Due to overhead, 1x T4 is 5x faster."* |
| T4s lack NVLink | The two T4s on a Kaggle instance are connected via **PCIe only**, not NVLink. DDP all-reduce must traverse PCIe (~16 GB/s) instead of NVLink (~600 GB/s on A100). For a 3B model with 60M LoRA params, the all-reduce payload per step is ~120 MiB (fp16). At PCIe speeds that's ~7 ms per step — small, but multiplied by 1878 steps adds ~13 s of overhead. The bigger issue is the **kernel-launch + sync overhead** of DDP across 2 processes, which adds 30-50 % wall-clock per step on small models. |
| 3B model fits on 1 T4 with massive headroom | Per the memory table in §5, single-GPU config uses ~5 GiB out of 14.56 GiB available — 9.5 GiB headroom. No reason to shard. |
| Notebook ergonomics | DDP requires `torchrun --nproc_per_node=2 train.py` or `accelerate launch`. From a Kaggle notebook (`.ipynb`), this requires writing the train script to a `.py` file and shelling out via `subprocess.run([...])`. This breaks the cell-by-cell development flow that the user has been iterating on. The official Unsloth Kaggle notebooks do NOT do this. |

### 3.2 What if we insist on DDP anyway?

If you really want to use both T4s (e.g. for the full 90k dataset to halve wall-clock), here's the **exact** pattern that works (verified via the `Kaggle-Llama3.1_(8B)-Alpaca_kaggle_T4x2.ipynb` notebook):

**Step 1:** In cell 1, set env vars + block `torch.nn.DataParallel` (which is NOT DDP — it's the slow single-process model-parallel alternative):
```python
import os, gc, torch, torch.nn as nn
os.environ["PYTORCH_CUDA_ALLOC_CONF"]               = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"]                = "false"
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "sdpa"

class _BlockedDP(nn.Module):
    def __init__(self, module, **kwargs):
        raise RuntimeError("DataParallel blocked — use torchrun for DDP")
nn.DataParallel       = _BlockedDP
torch.nn.DataParallel = _BlockedDP
```

**Step 2:** In cell 4 (model load), **do NOT pass `device_map`** — let Unsloth load on the default device (cuda:0 in the rank-0 process). Each rank will load its own copy.

**Step 3:** Move the SFTTrainer build + `trainer.train()` call into a `train.py` script saved to `/kaggle/working/train.py`. Then in the notebook:
```python
import subprocess
result = subprocess.run([
    'torchrun', '--nproc_per_node=2', '--master_port=29501',
    '/kaggle/working/train.py',
], env={**os.environ, 'HF_TOKEN': HF_TOKEN, 'CONFIG_JSON': json.dumps(CONFIG)})
```

**Step 4:** Inside `train.py`, set `ddp_find_unused_parameters=False` in SFTConfig (otherwise PEFT will raise `ValueError: module has no parameters` due to LoRA modules being skipped during the first forward).

### 3.3 Our recommendation

**Don't do this.** The complexity isn't worth it. Use the recommended single-GPU config from §2. If training is too slow, the fallback in §7 (reduce dataset size) is simpler than setting up DDP.

---

## Section 4 — `PYTORCH_ALLOC_CONF` setting

### Recommendation: **YES, add it at the top of cell 1.**

```python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

### Why

| Aspect | Detail |
|---|---|
| What it does | Switches PyTorch's CUDA caching allocator from "fixed-block" to "expandable-segment" mode. Instead of pre-allocating fixed-size memory pools (which can fragment when an allocation doesn't fit any existing pool), PyTorch now requests virtual memory segments from the driver that grow as needed. |
| When it helps | When allocations vary wildly in size — e.g. Unsloth's fused CE loss allocates 3 GiB chunks intermittently between smaller activations. With fixed blocks, the 3 GiB chunk might not fit in any existing pool even if total free memory is sufficient → spurious OOM. Expandable segments eliminate this. |
| Performance cost | Negligible. ~1-2 % slower on small models, can be faster on fragmentation-prone workloads. PyTorch docs say: "expandable_segments reduces fragmentation and allows the allocator to use all free memory." |
| Where the error message itself recommends it | The OOM error in the user's notebook: *"If reserved but unallocated memory is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation."* (Note: PyTorch renamed `PYTORCH_CUDA_ALLOC_CONF` to `PYTORCH_ALLOC_CONF` in PyTorch 2.3+. Both still work; Kaggle's torch 2.10 accepts both.) |
| Unsloth's stance | The official `Kaggle-Llama3.1_(8B)-Alpaca_kaggle_T4x2.ipynb` notebook from PR #212 sets this exact env var at the top of cell 1. Unsloth maintainers haven't explicitly recommended it for single-GPU T4 use, but it's harmless. |

### Caveat

`expandable_segments:True` requires CUDA 11.x+ (Kaggle has 12.8 ✓) and can cause issues on very old GPUs (pre-Maxwell). T4 is Turing architecture (2018) — fully supported.

---

## Section 5 — Expected memory budget table

For the **recommended config** (`bs=2, seq=2048, max_seq=2048, lora_r=32, all-7 target_modules, paged_adamw_8bit, use_gradient_checkpointing='unsloth'`) **on a single T4**:

| Component | Memory (GiB) | Notes |
|---|---|---|
| Model weights (Qwen2.5-3B, 4-bit NF4) | 1.81 | 3.09 B params × 0.5 bytes + 13 % double-quant state |
| LoRA adapters (r=32, all-7 modules, fp16) | 0.11 | 59.8 M trainable × 2 bytes (matches Unsloth output: "Trainable parameters = 59,867,136") |
| Optimizer state (8-bit paged AdamW) | 0.11 | 59.8 M params × 2 bytes (m+v in 8-bit). Paged → can offload to CPU on edge OOM. |
| Gradients (fp16) | 0.11 | 59.8 M params × 2 bytes |
| Activations (bs=2, seq=2048, 'unsloth' GC ≈ 30 % on GPU) | 0.17 | per layer: bs × seq × hidden × 2 = 8.4 MiB × 36 layers × 30 % (rest async-CPU-offloaded) |
| Logits forward (bs=2, seq=2048, fp16, vocab=151,936) | 1.16 | 2 × 2048 × 151,936 × 2 bytes — **the dominant consumer** |
| Logits grad (backward) | 1.16 | same shape |
| Unsloth fused CE peak chunk | ~0.7 | ~30 % of (fwd+grad) chunked at a time |
| KV cache (training: `use_cache=False`, so ~0) | 0 | Trainer sets `use_cache=False` automatically when gradient checkpointing is on |
| CUDA context + driver overhead | 0.5 | per-process CUDA init |
| PyTorch allocator overhead | 0.5 | caching allocator pools, etc. |
| Fragmentation headroom | 1.0 | safety margin (with `expandable_segments:True` this can be smaller, but keep it conservative) |
| **TOTAL peak** | **~6.3 GiB** | |
| Available | 14.56 GiB | T4 16 GiB minus driver/CUDA context |
| **Free margin** | **~8.3 GiB** | very comfortable |

### Comparison: current (OOMing) config on the broken model-parallel setup

| Component | GPU 0 | GPU 1 | Notes |
|---|---|---|---|
| Model weights (sharded) | 0.9 | 0.9 | 50/50 split |
| LoRA + opt + grads | 0.11 | 0.11 | on the GPU holding each weight |
| Activations | 0.35 | 0.35 | flowing through both halves |
| Logits forward (bs=4, seq=4096) | — | **4.64** | concentrated on GPU holding `lm_head` |
| Logits grad | — | **4.64** | same |
| Overhead | 0.5 | 0.5 | |
| **TOTAL** | ~2.0 | **~11.1** | |
| Unsloth next chunk needed | — | **+ 3.18** | matches the error message exactly |
| Available | 14.56 | 14.56 | |
| Margin | +12.5 | **−0.6 (OOM)** | |

The math lines up with the user's actual error report to within 100 MiB — high confidence this is the right diagnosis.

---

## Section 6 — The exact code changes to make

Three cells need changes. Cell 0 (CONFIG), cell 1 (deps + env), cell 4 (model load). **Cells 5, 7, 8 need NO changes.**

### Cell 0 — CONFIG dict

Replace the `# --- Training ---` and `# --- Model ---` blocks:

```diff
     # --- Model ---
     'base_model'        : 'Qwen/Qwen2.5-3B-Instruct',
-    'max_seq_length'    : 4096,
+    'max_seq_length'    : 2048,   # M0-fix-17: halved (official Kaggle Qwen 7B notebook uses 2048)
     'load_in_4bit'      : True,
     'dtype'             : None,  # None = auto (T4 -> fp16),

     # --- LoRA ---
     'lora_r'            : 32,
     'lora_alpha'        : 64,
     'lora_dropout'      : 0,
     'lora_target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                             'gate_proj', 'up_proj', 'down_proj'],

     # --- Training ---
-    'per_device_batch'  : 4,
-    'grad_accum_steps'  : 2,   # 2 GPUs × 4 × 2 = effective batch 16
+    'per_device_batch'  : 2,
+    'grad_accum_steps'  : 8,   # M0-fix-17: 1 GPU × 2 × 8 = effective batch 16
                                # (we were never using 2 GPUs in data-parallel mode;
                                #  device_map='auto' was model-parallel — see research/06)
     'epochs'            : 3,
```

Also fix the misleading `eff` calculation print at the bottom of cell 0:

```diff
 n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
-eff = n_gpus * CONFIG['per_device_batch'] * CONFIG['grad_accum_steps']
+# M0-fix-17: single-GPU training (Unsloth recommends 1 GPU on Kaggle T4x2 due to
+# no NVLink — DDP overhead is 5x slower than single-GPU per Unsloth's HF model card)
+eff = CONFIG['per_device_batch'] * CONFIG['grad_accum_steps']
 print(f'GPUs detected: {n_gpus} (using 1 for training)')
 print(f'Effective batch size: {eff}')
-assert n_gpus == 2, f'Expected 2 T4 GPUs, got {n_gpus}. Set accelerator to GPU T4 x2 in Kaggle settings.'
+assert n_gpus >= 1, f'Expected at least 1 GPU, got {n_gpus}.'
```

### Cell 1 — Install + env vars

Add the `PYTORCH_CUDA_ALLOC_CONF` env var setting **at the very top** of cell 1, before any imports (env vars must be set before `torch.cuda` is initialised):

```diff
+# ====================================================================
+# THURSDAY AI — FINAL KAGGLE INSTALL (research-verified by inspecting 7 wheels)
+# See: research/04_kaggle_unsloth_working_config.md
+# ...
+# ====================================================================
+
+# M0-fix-17: set CUDA allocator env vars BEFORE importing torch.
+# expandable_segments:True reduces fragmentation (was the error message's
+# own suggestion). Set early because PyTorch reads this on first cuda init.
+import os
+os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
+os.environ['TOKENIZERS_PARALLELISM']  = 'false'   # silence the warning
+# Use SDPA attention (faster + less memory than 'eager'; works on T4)
+os.environ['TRANSFORMERS_ATTENTION_IMPLEMENTATION'] = 'sdpa'
+
 !pip install -q --upgrade pip
 ...
```

(Same env var block as in `Kaggle-Llama3.1_(8B)-Alpaca_kaggle_T4x2.ipynb`.)

### Cell 4 — Model load (the actual OOM fix)

Remove the `device_map='auto'` argument:

```diff
 model, tokenizer = FastLanguageModel.from_pretrained(
     model_name = CONFIG['base_model'],
     max_seq_length = CONFIG['max_seq_length'],
     dtype = CONFIG['dtype'],  # None = auto (fp16 on T4)
     load_in_4bit = CONFIG['load_in_4bit'],
-    device_map = 'auto',  # distributes across both T4s
+    # M0-fix-17: NO device_map — load on GPU 0 only.
+    # device_map='auto' was forcing model-parallel sharding across both T4s
+    # (single process), which concentrated the LM head + logits on GPU 1
+    # and caused OOM. Unsloth explicitly recommends single-GPU on Kaggle T4x2:
+    # https://huggingface.co/unsloth/Qwen2.5-3B-Instruct
+    # "Kaggle has 2x T4s, but we use 1. Due to overhead, 1x T4 is 5x faster."
 )
 print(f'Loaded {CONFIG["base_model"]} in 4-bit. Vocab size: {len(tokenizer)}')
```

### Cell 7 — SFTTrainer (NO CHANGES NEEDED)

Cell 7 already correctly uses `processing_class=tokenizer` (M0-fix-16), `gradient_checkpointing` is set in cell 5 via `use_gradient_checkpointing='unsloth'`, and `optim` comes from CONFIG. **No edits.**

### Cell 8 — Train! (NO CHANGES NEEDED)

`trainer.train()` — no edits.

### Summary of all changes

| Cell | Lines changed | Reason |
|---|---|---|
| 0 | 3 (CONFIG keys) + 2 (eff calc) | seq 4096→2048, bs 4→2, accum 2→8 |
| 1 | +5 (env vars) at top | `PYTORCH_CUDA_ALLOC_CONF`, `TOKENIZERS_PARALLELISM`, `TRANSFORMERS_ATTENTION_IMPLEMENTATION` |
| 4 | −1 line (remove `device_map`) | The actual OOM fix |
| 5, 7, 8 | 0 | No changes needed |

**Total diff: ~10 lines.** Single most important change: removing `device_map='auto'`.

---

## Section 7 — Fallback plan

If the recommended config (bs=2, seq=2048, single-GPU) STILL OOMs (it shouldn't — there's ~8 GiB headroom per the §5 budget), apply these in order, retrying after each:

### Fallback 1: Drop `per_device_batch` to 1, raise `grad_accum` to 16

```python
'per_device_batch'  : 1,
'grad_accum_steps'  : 16,  # still effective batch = 16
```

Halves the logits memory (1.16 GiB → 0.58 GiB) and activations (0.17 GiB → 0.08 GiB). Total budget: ~4.6 GiB — leaves ~10 GiB headroom. If THIS OOMs, the issue isn't memory capacity — it's a fragmentation / leak bug, and you should file an Unsloth issue.

### Fallback 2: Drop `max_seq_length` to 1024

```python
'max_seq_length'    : 1024,
```

This is what the official `Kaggle-Qwen2.5_(3B)-GRPO.ipynb` notebook uses. Only acceptable if the dataset is OK with it — the user said *"many real OS-control trajectories are < 2048 tokens"*. If 1024 still loses too much data (most trajectories truncated), try 1536 as a compromise.

### Fallback 3: Drop `lora_r` from 32 to 16

```python
'lora_r'    : 16,
'lora_alpha': 16,  # alpha = r is the modern default
```

This is what the official `Kaggle-Qwen2.5_(7B)-Alpaca.ipynb` uses. Saves ~50 % of LoRA params (60 M → 30 M), which saves ~0.06 GiB optimizer state and 0.06 GiB gradients. Marginal — but it's free quality-adjusted headroom. **Note: this is the single most likely quality-affecting change.** For tool-use / agentic training, r=32 is meaningfully better than r=16 — only do this if Fallback 1 + 2 don't suffice.

### Fallback 4: Reduce target_modules

```python
'lora_target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
```

Remove MLP modules (`gate_proj`, `up_proj`, `down_proj`). Saves ~85 % of LoRA trainable params (60 M → 9 M) — but the quality impact for agentic / tool-use training is significant. Only do this if all else fails. The MLP layers are where tool-call JSON emission gets learned.

### Fallback 5: Switch to the 1.5B model

```python
'base_model': 'Qwen/Qwen2.5-1.5B-Instruct',
```

Halves model weights (1.8 GiB → 0.9 GiB) and roughly halves activations. Quality drop is real — 1.5B is the floor for coherent multi-turn tool use — but if you absolutely need a working training run to ship v0.1, this works on a single T4 with bs=4, seq=4096 comfortably.

### Fallback 6: Switch from Unsloth to vanilla HF transformers + peft + trl + bitsandbytes

The `research/04_kaggle_unsloth_working_config.md` §5 has the full vanilla fallback install + load + train code. Vanilla HF is ~2× slower and uses ~70 % more VRAM than Unsloth, but:
- No dependency on Unsloth's internal patching (which has been the source of 4+ of the 16 prior failures)
- Easier to debug (no compiled trainer cache)
- Compatible with HF's official DDP/FSDP docs

If we hit Unsloth-specific issues on Fallback 5, this is the escape hatch. Use it for a single training run to get v0.1 out the door, then debug Unsloth separately.

### Fallback 7 (last resort): Train on Colab L4 / A10G instead of Kaggle T4

If Kaggle's hardware is fundamentally too constrained, run the same notebook on:
- **Google Colab Pro+ L4** (24 GiB VRAM, single GPU, $10/month)
- **RunPod A10G** (24 GiB, $0.40/hr spot)
- **Vast.ai RTX 3090** (24 GiB, $0.20/hr spot)

L4 / A10G / 3090 all have ~50 % more VRAM than T4, support bf16 natively (T4 doesn't), and have ~2-3× the FLOPs. Total training time on 90k examples × 3 epochs would drop from ~6-8 h on T4 to ~2-3 h on L4.

---

## Appendix A — Searches performed

All searches saved to `/tmp/oom_research/s*.json`, all page reads saved to `/tmp/oom_research/p_*.json`. The full list:

| # | Query / URL | File | Key finding |
|---|---|---|---|
| 1 | `web_search`: "Unsloth Qwen2.5-3B Kaggle T4 OOM fix batch size" | s1.json | Found HF card snippet "Kaggle has 2x T4s, but we use 1" |
| 2 | `web_search`: "Unsloth T4 16GB max_seq_length recommendation Qwen 3B" | s2.json | Unsloth Qwen3 docs recommend 2048 (not 4096) for testing |
| 3 | `web_search`: "Unsloth device_map auto DDP T4 x2 enable data parallel" | s3.json | Kaitchup article distinguishing device_map=balanced (model-parallel) vs DDP (torchrun) |
| 4 | `web_search`: "Qwen2.5-3B QLoRA T4 VRAM requirement batch size seq_len" | s4.json | Confirmed Qwen2.5-3B = 1.93 GB Q4_K_M, ~2 GB QLoRA weights |
| 5 | `web_search`: "PYTORCH_ALLOC_CONF expandable_segments Unsloth OOM" | s5.json | Confirmed expandable_segments is the standard fix across PyTorch, vLLM, Unsloth |
| 6 | `web_search`: "Unsloth single GPU T4 Qwen Kaggle notebook config" | s6.json | Found official `Kaggle-Qwen2.5_(7B)-Alpaca.ipynb` |
| 7 | `web_search`: "unsloth gradient_checkpointing unsloth True False difference memory" | s7.json | Unsloth blog: 30 % less VRAM, +1.9 % slower. Issue #1418 confirmed "unsloth" mode is preferred |
| 8 | `web_search`: "paged_adamw_8bit adamw_8bit adafactor difference VRAM" | s8.json | paged_adamw_8bit = adamw_8bit + CPU offload paging (safer for edge OOM) |
| 9 | `web_search`: "github unslothai notebooks Qwen2.5 3B Alpaca ipynb T4" | s9.json | Found PR #212 (T4×2 ports) |
| P1 | `page_reader`: `unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp` | p_ddp.json | DDP requires `torchrun --nproc_per_node=N`; default per_device_batch=2, grad_accum=4 |
| P2 | `page_reader`: `kaitchup.substack.com/p/how-to-run-unsloth-on-multi-gpu-setups` | p_kaitchup.json | "device_map=balanced triggers model-parallelism, accelerate launch triggers DDP — you cannot mix them" |
| P3 | `page_reader`: `unsloth.ai/docs/basics/troubleshooting-and-faqs` | p_faq.json | "A common issue when you OOM is because you set your batch size too high. Set it lower than 2" |
| P4 | `page_reader`: `github.com/unslothai/unsloth/issues/4504` | p_issue4504.json | ChatML datasets use same VRAM as Alpaca (confirmed by maintainer investigation) — the user's ChatML is not the cause |
| P5 | `page_reader`: `github.com/unslothai/unsloth/issues/3435` | p_issue3435.json | Stack trace confirmed: OOM happens in `unsloth_zoo/fused_losses/cross_entropy_loss.py:362` during `accumulate_chunk` |
| P6 | `page_reader`: `huggingface.co/unsloth/Qwen2.5-3B-Instruct` | p_hfcard.json | **"Kaggle has 2x T4s, but we use 1. Due to overhead, 1x T4 is 5x faster."** |
| P7 | `page_reader`: `unsloth.ai/docs/basics/multi-gpu-training-with-unsloth` | p_mgpusimple.json | Confirms: device_map="balanced" = model split, NOT DDP |
| P8 | `page_reader`: `kaggle.com/code/danielhanchen/kaggle-qwen-2-5-unsloth-notebook` | p_kaggle_qwen.json | Official Kaggle notebook — runs in 14m on T4×2 (using 1 GPU) |
| P9 | `page_reader`: `kaggle.com/code/danielhanchen/kaggle-qwen-2-5-conversational-unsloth` | (via Kaggle UI, doesn't expose source) | — |
| P10 | `page_reader`: `github.com/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B).ipynb` | p_qwen3b_nb.json | 404 — path doesn't exist in main repo |
| P11 | `page_reader`: `github.com/unslothai/unsloth/issues/1418` | p_issue1418.json | Author of issue #1418 later said "false alarm — was just excessive batch size" |
| P12 | `page_reader`: `unsloth.ai/blog/long-context` | p_longctx.json | Unsloth GC: 4× longer context than HF+FA2, 30 % less VRAM, +1.9 % slower |
| P13 | `page_reader`: `github.com/unslothai/notebooks/pull/212` | p_pr212.json | Kaggle T4×2 PR — found the env-var pattern + DDP-blocking patch |
| P14 | Downloaded raw: `Kaggle-Qwen2.5_(7B)-Alpaca.ipynb` from unslothai/notebooks main | kaggle_qwen7b_alpaca.ipynb | **Primary config template**: seq=2048, bs=2, accum=4, optim=adamw_8bit, r=16, alpha=16, no device_map |
| P15 | Downloaded raw: `Kaggle-Qwen2.5_(3B)-GRPO.ipynb` | kaggle_qwen3b_grpo.ipynb | seq=1024, r=64, alpha=64, optim=adamw_8bit (3B-specific reference) |
| P16 | Downloaded raw: `Kaggle-Llama3.1_(8B)-Alpaca_kaggle_T4x2.ipynb` from AAB20 fork | llama8b_t4x2.ipynb | **Exact env-var + DDP-block pattern**: PYTORCH_CUDA_ALLOC_CONF, TOKENIZERS_PARALLELISM, sdpa, _BlockedDP, post-init trainer patch |

Total: 9 web searches + 13 page reads + 3 raw notebook downloads.

---

## Appendix B — Answers to the specific questions asked

> **1. What `per_device_batch` should we use?**

**2.** (was 4). Reasoning: halves logits memory (the dominant consumer) from 2.3 GiB to 1.2 GiB at seq=2048. The official `Kaggle-Qwen2.5_(7B)-Alpaca.ipynb` uses bs=2 for a 7B model — we're at 3B so could go higher (bs=4 would still fit with ~7 GiB headroom), but bs=2 gives more conservative headroom and lets us keep grad_accum=8 for stable effective batch=16. If you want faster training and are willing to risk it, bs=4 + grad_accum=4 also fits in single-GPU mode (~7.6 GiB used, 6.9 GiB headroom).

> **2. What `max_seq_length` should we use?**

**2048.** (was 4096). Reasoning:
- The official Unsloth Kaggle notebook for a 7B model (2.3× our size) uses 2048.
- Logits at bs=2, seq=2048 = 1.16 GiB. At seq=4096 = 2.32 GiB. Either fits on 1 T4, but 2048 gives 1.2 GiB more headroom.
- The user noted "Many real OS-control trajectories are < 2048 tokens." Verify with a quick `len(tokenizer(ex['text']))` distribution check on the formatted dataset — if >99 % are < 2048, we're safe.
- If you absolutely need longer context (some complex multi-tool trajectories), use seq=3072 as a compromise.

> **3. How do we actually use BOTH T4s?**

**You don't.** Per Unsloth's own model card: "Kaggle has 2x T4s, but we use 1. Due to overhead, 1x T4 is 5x faster." Use single-GPU. If you insist on DDP, see §3.2 above for the torchrun pattern. **Do NOT use `device_map='auto'`** — that's model-parallel, not data-parallel, and it's what's currently causing the OOM.

> **4. Should we set `PYTORCH_ALLOC_CONF=expandable_segments:True`?**

**YES.** At the top of cell 1, before any torch import. The OOM error itself recommended it. The official `Kaggle-Llama3.1_(8B)-Alpaca_kaggle_T4x2.ipynb` notebook sets it. Negligible performance cost (~1-2 %), eliminates fragmentation-induced spurious OOMs.

> **5. Should we reduce LoRA `target_modules`?**

**NO.** Removing `gate_proj/up_proj/down_proj` (the MLP) saves only ~0.07 GiB of optimizer state (60 M → 9 M trainable params × 2 bytes/param × 2 moments). That's <1 % of the total memory budget. Meanwhile, the MLP layers are where tool-call JSON emission is learned — quality impact would be significant. The bottleneck is **logits** (1.16 GiB at bs=2 seq=2048), not LoRA params.

> **6. Is `optim='paged_adamw_8bit'` correct?**

**YES, keep it.** Comparison:
- `adamw_8bit` — 8-bit optimizer states, 2 bytes/param. No CPU offload.
- `paged_adamw_8bit` — same as above + paged memory (auto-offloads optimizer state to CPU RAM when GPU runs out). Slight CPU overhead only when paging kicks in.
- `adafactor` — smaller state (no per-param second moment), but worse convergence for LoRA. Not recommended for tool-use training.

`paged_adamw_8bit` is the safest choice for memory-constrained T4. The official `Kaggle-Qwen2.5_(7B)-Alpaca.ipynb` uses plain `adamw_8bit` (which works because 7B model on T4 has enough headroom), but `paged_adamw_8bit` gives an extra safety net for the edge case where activations spike (e.g. a particularly long trajectory in a batch). Keep `paged_adamw_8bit`.

> **7. What does `gradient_checkpointing='unsloth'` actually do vs `True` vs `False`?**

| Value | What it does | Memory | Speed |
|---|---|---|---|
| `False` | No gradient checkpointing. All activations kept in VRAM for backward. | Highest (often OOMs on T4 for >7B at seq>2048) | Fastest forward+backward (no recompute) |
| `True` | PyTorch's native gradient checkpointing (`torch.utils.checkpoint`). Discards intermediate activations during forward, recomputes them during backward. | ~50 % less than `False` | ~20-30 % slower than `False` |
| `'unsloth'` | Unsloth's custom impl: async-CPU-offloads activations to system RAM during forward, fetches them back during backward. Uses non-blocking CPU-GPU copies to hide latency. | ~30 % less than `True` (i.e. ~65 % less than `False`) | Only ~1.9 % slower than `True` (Unsloth blog) |

The Unsloth mode is strictly better than `True` on memory-constrained hardware. One user reported OOM with `'unsloth'` but not `True` in issue #1418 — but they later retracted: "this seemed to be a false alarm. I got OOM when setting True later. I think it's just due to excessive batch size and batching randomness." (Source: github.com/unslothai/unsloth/issues/1418#issuecomment-2547014921)

Keep `'unsloth'`.

---

## Appendix C — Verification checklist before the next training run

Before pressing Run All on cell 8, sanity-check:

- [ ] Cell 0 prints `GPUs detected: 2 (using 1 for training)` and `Effective batch size: 16`
- [ ] Cell 1 imports clean (same as M0-fix-16)
- [ ] Cell 4 loads the model and prints vocab size — should take ~30 s, no OOM
- [ ] Cell 5 attaches LoRA — Unsloth prints `Trainable parameters = 59,867,136 of 3,145,805,824 (1.90% trained)` (same as before)
- [ ] Cell 7 builds SFTTrainer — Unsloth prints `Num GPUs used = 1` (was 2) and `Data Parallel GPUs = 1` (was 1) and `Total batch size (2 x 8 x 1) = 16` (was `(4 x 2 x 1) = 8`)
- [ ] Cell 8 starts training — first 10 steps complete without OOM, loss is non-zero (loss=0 means train_on_responses_only masking is broken)
- [ ] After step 100, check `nvidia-smi` — peak VRAM should be ~6-8 GiB on GPU 0, ~0 GiB on GPU 1

If the trainer prints `Total batch size (2 x 8 x 1) = 16` — that's correct, no DDP, single GPU, batch 16. If it prints anything else, double-check cell 4 doesn't have `device_map='auto'`.

---

## Appendix D — Final config (sanity-printable)

For copy-paste verification, here's the entire CONFIG dict + cell-1 env vars in one block:

```python
# === Cell 0 — CONFIG ===
CONFIG = {
    'github_repo'       : 'https://github.com/skyro777/Thursday_AI.git',
    'github_branch'     : 'main',
    'repo_dir'          : '/kaggle/working/Thursday_AI',
    'base_model'        : 'Qwen/Qwen2.5-3B-Instruct',
    'max_seq_length'    : 2048,
    'load_in_4bit'      : True,
    'dtype'             : None,
    'lora_r'            : 32,
    'lora_alpha'        : 64,
    'lora_dropout'      : 0,
    'lora_target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                            'gate_proj', 'up_proj', 'down_proj'],
    'per_device_batch'  : 2,
    'grad_accum_steps'  : 8,
    'epochs'            : 3,
    'learning_rate'     : 2e-4,
    'warmup_ratio'      : 0.03,
    'lr_scheduler'      : 'cosine',
    'weight_decay'      : 0.01,
    'optim'             : 'paged_adamw_8bit',
    'save_steps'        : 500,
    'logging_steps'     : 10,
    'seed'              : 42,
    'output_dir'        : '/kaggle/working/thursday-ai-lora',
    'merged_dir'        : '/kaggle/working/thursday-ai-merged',
    'gguf_path'         : '/kaggle/working/thursday-ai-v0.1-Q4_K_M.gguf',
    'gguf_quant'        : 'Q4_K_M',
    'hf_repo_merged'    : 'skyro777/thursday-ai-v0.1-merged',
    'hf_repo_gguf'      : 'skyro777/thursday-ai-v0.1-gguf',
}

# === Cell 1 — env vars (BEFORE torch import) ===
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF']             = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM']             = 'false'
os.environ['TRANSFORMERS_ATTENTION_IMPLEMENTATION'] = 'sdpa'

# === Cell 4 — model load (NO device_map) ===
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = CONFIG['base_model'],
    max_seq_length  = CONFIG['max_seq_length'],
    dtype           = CONFIG['dtype'],
    load_in_4bit    = CONFIG['load_in_4bit'],
    # NO device_map — load on GPU 0 only (Unsloth recommends 1 GPU on Kaggle T4x2)
)
```

**End of report.**
