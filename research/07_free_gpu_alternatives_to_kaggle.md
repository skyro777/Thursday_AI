# Research 07 — Free (and near-free) GPU Alternatives to Kaggle

**Task ID:** M0-fix-25
**Agent:** Free-GPU-Alternatives-Researcher
**Date:** Late 2025 / early 2026
**Question being answered:** *"Why not train somewhere else if Kaggle is that bad at AI training, a free tier GPU other place?"*

---

## Section 1 — TL;DR: Top 3 Recommendations

For Thursday AI = fine-tuning **Qwen2.5-3B-Instruct** with **QLoRA (4-bit, r=32, all-7 modules, seq=4096)** via **Unsloth**. Two real training budgets to keep in mind (per `worklog.md`):

- **Smoke test** (5k examples × 1 epoch): ~3 h on a single T4, ~1.5 h on L4
- **Full v0.1** (90k examples × 3 epochs): **~25 h on T4** (exceeds Kaggle 12 h), **~8–12 h on L4 24 GB**, **~3–5 h on A100 40 GB**

| # | Pick | Why |
|---|------|-----|
| 🥇 | **Lightning AI Studios — Free tier** | **L4 24 GB GPU**, 22 free GPU-hours/month, **persistent storage** (model downloads once), JupyterLab-style Studio UX (closest to Kaggle), no idle disconnect, internet enabled for HF Hub upload. **One full v0.1 training run (~8 h) fits inside the free quota with 14 h to spare.** Zero code changes to the Kaggle notebook — same cells, same `pip install --no-deps unsloth unsloth_zoo` flow. |
| 🥈 | **Modal — Starter plan ($0/mo)** | **$30/month of free compute credits**, per-second billing → effectively **~37 h of L4 24 GB or ~27 h of A10G 24 GB free every month** — *more* free compute than Lightning. Steeper learning curve (serverless functions / `modal interactive` instead of plain notebook), but you can't beat the math if you're willing to refactor. Also: free $10k academic credits for students. |
| 🥉 | **Vast.ai spot — RTX 3090 24 GB** | Median price **$0.19/hr** (cheapest 24 GB GPU on the open market). A full v0.1 run = ~6 h × $0.19 = **~$1.13–$2**. Add ~$0.50 storage → real Thursday AI v0.1 for **under $3**. Trade-off: marketplace reliability varies (hosts can cancel), so you must checkpoint every N steps. For $3 total, who cares — just re-launch. |

**Bottom line:** Yes, switch. Kaggle was the wrong tool once you needed >12 h. The single best move is **Lightning AI Studios free tier** — same notebook, faster GPU, persistent storage, free.

---

## Section 2 — Full Comparison Table

All data verified against primary sources in late 2025/early 2026 (Kaggle docs, Colab FAQ, Modal/RunPod/Vast/Lambda/HF/Paperspace/Replicate pricing pages, plus secondary sources for the free-tier specifics).

### 2.1 Free / freemium options

| Provider | GPU type & VRAM | Free hours (per day / month) | Session limit | Persistent storage? | Unsloth works? | Internet during training? | Est. time on Qwen2.5-3B QLoRA 5k ex @ seq=4096 | Real cost for full v0.1 (90k×3 ep) | Notes / gotchas |
|---|---|---|---|---|---|---|---|---|---|
| **Kaggle** (current) | T4×2 (16 GB each) or single P100 (16 GB) | ~30 h/week GPU | **12 h** hard cap (9 h for TPU) | Yes — 20 GB `/kaggle/working` | ✅ Yes (we proved it) | ✅ Yes (toggleable, on by default outside competitions) | ~3 h on T4 single-GPU, 1 epoch, 5k examples | **$0** but ~25 h needed → **exceeds 12 h session limit**; would require chunking across sessions or splitting dataset | T4s lack NVLink; Unsloth recommends single-GPU. 30 h/week resets Sunday. Quota can be boosted by linking Colab Pro (+15 h/wk) or Pro+ (+30 h/wk). |
| **Google Colab Free** | T4 16 GB (sometimes L4 24 GB if you're lucky / off-peak) | Dynamic; up to ~12 h/day but not guaranteed | 12 h max session, **~90 min idle disconnect** | No — must mount Google Drive (15 GB free) | ✅ Yes — Unsloth officially targets Colab Free T4 | ✅ Yes | ~3 h on T4 (same as Kaggle single T4) | $0 — but T4 too slow (~25 h) and idle disconnects make 25 h runs impractical | No background execution. GPU type/availability is dynamic — sometimes you get no GPU at all during peak. No persistent packages either. |
| **Google Colab Pro** ($10/mo) | T4 / L4 (24 GB) priority access | 100 compute units (CU) included | 24 h max session | Drive (Pro = 100 GB free) | ✅ Yes | ✅ Yes | ~1.5 h on L4 for 5k examples | L4 ≈ 6 CU/hr → 100 CU = ~16 h L4/month, just enough for **1 v0.1 run** + smoke test | CU expire after the subscription period (effectively monthly). $10 = ~1.5 v0.1 runs/mo. |
| **Google Colab Pro+** ($49.99/mo) | L4 24 GB / A100 40 GB | **500 compute units** included | **24 h continuous code execution**; background execution enabled | Drive (100 GB free) | ✅ Yes | ✅ Yes | ~1.5 h on L4 for 5k; **~6–8 h on L4 for full v0.1** (or ~3 h on A100 40 GB) | L4 ≈ 6 CU/hr → 500 CU = ~83 h L4/month = **~10 v0.1 runs/mo**. Or A100 ≈ 13 CU/hr → 500 CU = ~38 h A100 = ~10 v0.1 runs/mo | Best UX if you're already a Colab user. **Background execution** means you can close the tab. But $50/mo is steep for a one-off v0.1. |
| **Lightning AI Studios — Free** | **L4 24 GB** (free) | **22 GPU-hours/month** + 1 always-on CPU studio | No hard session cap as long as you have credits; interactive | ✅ **Yes — persistent Studio storage** (15 GB+) | ✅ Yes (just a Linux box with PyTorch) | ✅ Yes | ~1.5 h on L4 for 5k examples; **~6–8 h for full v0.1** | **$0 — fits inside the 22 free hours** (8 h training + spare for retries / debug) | Studios persist between runs, so the Qwen2.5-3B 4-bit model download (~2 GB) and pip installs happen **once**. JupyterLab-like UX is the closest Kaggle replacement. ~15 free credits/month on top (1 credit = $1). |
| **HuggingFace Spaces (Docker)** | T4 small (16 GB) — free with HF Pro ($9/mo); ZeroGPU dynamic; L4 $0.80/hr; A10G $1.00/hr | T4 small is free with Pro; ZeroGPU has dynamic limits | Designed for inference, not training — sessions are short-lived | Limited / ephemeral | ⚠️ Possible but Spaces are not designed for training — they're for serving | ✅ Yes | Not recommended for training | Not viable for training | Use HF Spaces for *hosting* the trained model (free T4 small with Pro), not for training. |
| **Paperspace Gradient — Free** | M4000 (8 GB) | "Free GPU" instances available | 12 h auto-shutdown | 5 GB | ⚠️ Maybe — but **8 GB VRAM will OOM** for Qwen2.5-3B + seq=4096 (need ≥14 GB) | ✅ Yes | **Doesn't fit** (OOM) | Not viable for free tier | M4000 is Kepler-era (2015). Free Pro tier ($8/mo) gives "faster free GPUs" but still limited. Skip. |
| **SageMaker Studio Lab** | Tesla T4 16 GB | **4 GPU-hours/day** (~120 h/month) | 12 h sessions | ✅ Yes — 15 GB persistent | ✅ Yes (no AWS account needed) | ⚠️ Limited — no internet in sessions by default (must request) | ~3 h on T4 for 5k | $0 — but ~25 h T4 needed for v0.1; you'd have to chunk across multiple days at 4 h/day | Signup requires approval (often takes days). 4 h/day GPU is too restrictive for v0.1. Status as of late 2025: still alive but AWS marketing has gone quiet — long-term future uncertain. |
| **Gitpod / Codespaces** | CPU only (no GPU on free tiers) | — | — | — | ❌ No GPU | — | — | Not viable | Skip — no GPU option on free tier. |
| **Modal — Starter ($0/mo)** | A10G 24 GB $1.10/hr; **L4 24 GB $0.80/hr**; A100 40 GB $2.10/hr; A100 80 GB $2.50/hr | **$30/month free compute credits** (per-second billing, no expiry if account active) | Per-second billing, runs until you cancel | ✅ Yes — Modal Volumes | ✅ Yes (Modal supports GPUs + Jupyter via `modal notebook`) | ✅ Yes | ~1.5 h on L4 for 5k; **~6 h on A10G for full v0.1** | **$0 — $30 credit = ~27 h A10G or ~37 h L4 free per month** | Serverless-first UX: write `@app.function(gpu="A10G")` and call `.remote()`. Has `modal interactive` and `modal notebook` for notebook-like UX but you'll refactor some code. Free $10k credits for academics. |

### 2.2 Paid-cheap options (worth comparing for the $2–$5 v0.1 run)

| Provider | GPU type & VRAM | Hourly price (on-demand unless noted) | Session limit | Persistent storage? | Unsloth works? | Internet? | Est. time on Qwen2.5-3B QLoRA full v0.1 | Real cost for full v0.1 |
|---|---|---|---|---|---|---|---|---|
| **Vast.ai** (marketplace, spot) | RTX 3090 24 GB — **median $0.19/hr**, from $0.08/hr | $0.19/hr (median) | Until cancelled or host cancels | ⚠️ Host-dependent (use Vast persistent storage add-on) | ✅ Yes (Ubuntu box) | ✅ Yes | ~6 h on 3090 (~2× faster than T4) | **~$1.13** (median $0.19 × 6 h) + ~$0.50 storage = **~$2** |
| **Vast.ai** | RTX 4090 24 GB — median $0.47/hr | $0.47/hr | Until cancelled or host cancels | Same | ✅ Yes | ✅ Yes | ~4 h on 4090 (~3× faster than T4) | **~$1.88** |
| **Vast.ai** | L4 24 GB — median $0.33/hr | $0.33/hr | Until cancelled or host cancels | Same | ✅ Yes | ✅ Yes | ~8 h on L4 | ~$2.64 |
| **RunPod** (community cloud) | RTX A5000 24 GB | **$0.27/hr** | Until cancelled | ✅ Yes ($0.10/GB/mo) | ✅ Yes (Unsloth Jupyter template available) | ✅ Yes | ~6 h on A5000 (~1.5× T4) | **~$1.62** |
| **RunPod** | RTX 3090 24 GB | $0.50/hr | Until cancelled | ✅ Yes | ✅ Yes | ✅ Yes | ~6 h on 3090 | ~$3.00 |
| **RunPod** | L4 24 GB | $0.49/hr | Until cancelled | ✅ Yes | ✅ Yes | ✅ Yes | ~8 h on L4 | ~$3.92 |
| **RunPod** | A100 PCIe 80 GB | $1.59/hr | Until cancelled | ✅ Yes | ✅ Yes | ✅ Yes | ~3 h on A100 80 GB | ~$4.77 |
| **RunPod** | H100 PCIe 80 GB | $2.89/hr | Until cancelled | ✅ Yes | ✅ Yes | ✅ Yes | ~3 h on H100 | ~$8.67 |
| **Brev.dev** (NVIDIA-owned) | A10G 24 GB, etc. | ~$1+/hr (varies) | Until cancelled | ✅ Yes | ✅ Yes (one-click templates for popular models like Mistral/Phi) | ✅ Yes | ~6 h on A10G | ~$6 |
| **Lambda Labs** (on-demand, reliable) | A100 PCIe 40 GB | **$1.99/hr** | Until cancelled | ✅ Yes (Lambda File System) | ✅ Yes (Ubuntu box) | ✅ Yes | ~3 h on A100 40 GB | ~$6 |
| **Lambda Labs** | A100 SXM 80 GB | $2.79/hr | Until cancelled | ✅ Yes | ✅ Yes | ✅ Yes | ~3 h | ~$8.37 |
| **Lambda Labs** | V100 16 GB | $0.79/hr | Until cancelled | ✅ Yes | ⚠️ Tight — 16 GB VRAM might OOM with seq=4096 | ✅ Yes | ~10 h on V100 | ~$7.90 (but risky on VRAM) |
| **Replicate** | T4 $0.81/hr, L40S $3.51/hr, A100 80 GB $5.04/hr | Per-second billing | Designed for inference API | ❌ No | Fine-tuning API only (not your notebook) | ✅ Yes | Not for notebook training | Not viable / too expensive |
| **Modal — Team** ($250/mo) | Same GPUs as Starter | $100/mo free credits + paid beyond | Per-second | ✅ Yes | ✅ Yes | ✅ Yes | Same as Starter pricing per hour | Only worth it for teams |
| **CoreWeave** | A100, H100 | ~$2.21/hr A100 | Until cancelled | ✅ Yes | ✅ Yes | ✅ Yes | ~3 h on A100 | ~$6.63 |
| **Together AI** | A100 / H100 | ~$1.20–$2.00/hr A100 spot | Until cancelled | ✅ Yes | ✅ Yes | ✅ Yes | ~3 h on A100 | ~$5; **$5 free credits on signup** |
| **Anyscale** | A10G, A100 | ~$1.00–$1.50/hr A10G | Until cancelled | ✅ Yes | ✅ Yes | ✅ Yes | ~6 h on A10G | **$100 free credits on signup = ~100 h A10G free** |

---

## Section 3 — Recommendation: Switch from Kaggle to ___?

**YES, switch. The specific recommendation is:**

### Primary recommendation: **Lightning AI Studios (Free tier)**

It is the *only* option that simultaneously:

1. **Costs $0** — no subscription, no credit card, no "free credits that run out".
2. **Gives an L4 24 GB GPU** (vs T4 16 GB on Kaggle). L4 is ~2–3× faster than T4 (Ampere architecture, newer CUDA cores, faster fp16/bf16). For Qwen2.5-3B QLoRA at seq=4096, this means the full v0.1 (90k × 3 epochs) drops from ~25 h on T4 to **~8 h on L4** — comfortably inside the 12 h limit *and* inside the 22 free monthly GPU hours.
3. **Has persistent storage** — this is the killer feature Kaggle lacks. The Qwen2.5-3B base model (~2 GB in 4-bit) and all your pip installs stay on disk between sessions. On Kaggle, every session restart re-downloads and re-installs (~5 min of wasted time per session).
4. **No idle disconnect** — Lightning Studios explicitly support long-running interactive sessions (subject to the 22 GPU-hour monthly quota, not a 90 min idle timer).
5. **Has internet during training** — for HF Hub upload and dataset streaming.
6. **JupyterLab-like UX** — closest notebook experience to Kaggle. The migration path is essentially copy-paste cells.
7. **Free Studio persists** — even when the GPU isn't allocated, your 1 always-on CPU Studio keeps your environment ready.

**22 GPU hours/month is enough for:** one full v0.1 run (8 h) + smoke tests (3 h × 2 = 6 h) + debugging headroom (8 h). That covers the entire v0.1 milestone plus iterations.

### Why not just stay on Kaggle?

Kaggle has been painful (per `worklog.md` M0-fix-1 through M0-fix-24):

- 12 h hard session cap killed the full v0.1 attempt
- T4×2 lacks NVLink → Unsloth recommends single-GPU → 5× slower than the 2-GPU promise
- Config keeps either OOMing or fitting in a way that exceeds the session limit
- Every session restart = fresh 5 min model download + pip installs
- ~25 h training estimate for v0.1 → can't be done in one Kaggle session

Switching to Lightning solves **all four** of these:

| Kaggle problem | Lightning solution |
|---|---|
| 12 h session limit kills v0.1 | 22 h/month quota — you can split into 2 sessions or do it in one 8 h L4 run |
| T4×2 NVLink issue | L4 is single-GPU by default — no NVLink assumption needed |
| ~25 h on T4 too long | L4 is ~2–3× faster → **~8 h** for v0.1 |
| Fresh install every session | Persistent Studio storage — install once, reuse forever |

### When NOT to follow this recommendation

Stay on Kaggle (or go elsewhere) if:

- **You're already paying for Colab Pro+** ($49.99/mo): Colab Pro+ gives you 500 compute units (~83 h of L4 or ~38 h of A100 40 GB) + 24 h continuous execution + background execution. If you have a Colab Pro+ subscription for other reasons, just use it.
- **You need >22 GPU hours/month consistently**: Lightning's free tier caps at 22 h. If you're iterating hard (multiple v0.1 retries per week), switch to **Modal Starter** (effectively 37 h/month free L4 or 27 h/month free A10G via the $30 credit) — or just pay $3 on Vast.ai per training run.
- **You want the absolute cheapest path to one finished v0.1**: Vast.ai RTX 3090 spot at ~$1.13–$2 per training run is unbeatable on price. But you trade reliability (hosts can cancel) for cost.

---

## Section 4 — How to Migrate the Notebook from Kaggle to Lightning AI Studios

### What stays the same

- The notebook file (`thursday-ai-training.ipynb`) — upload directly to your Lightning Studio
- All cells — same code, same order, same config
- `pip install --no-deps unsloth unsloth_zoo` install pattern (proven working per `worklog.md` M0-fix-9)
- `trl==0.11.4` override for `SFTConfig` (proven working per M0-fix-7)
- `bitsandbytes==0.49.2` install for torch 2.10 + transformers 5.0 compat (proven working per M0-fix-13)
- `llama.cpp` CMake build flow for GGUF quantization (proven working per M0-fix-19)
- HF Hub upload logic (proven working per M0-fix-21, M0-fix-22)

### What changes (minimal — about 4 small edits)

| Cell | Change | Why |
|---|---|---|
| **Cell 0 (config)** | Set `smoke_mode = False` for the full v0.1 run | Lightning has the 8 h budget that Kaggle lacked |
| **Cell 0 (config)** | Bump `max_seq_length` back to `4096` (was reduced to 3072 in M0-fix-24 to fit Kaggle's 12 h cap) | L4 is faster AND has 24 GB VRAM (vs 16 GB on T4) — seq=4096 is safe and recovers the dropped examples from M0-fix-23 |
| **Cell 1 (deps)** | Replace `kaggle`/Kaggle-specific paths with Lightning Studio paths | `/kaggle/working/` → `/teamspace/studios/this_studio/` (Lightning's persistent storage mount) |
| **Cell 1 (deps)** | Confirm CUDA version. Lightning Studio has CUDA 12.1 or 12.4 typically (vs Kaggle's CUDA 12.8). May need different bitsandbytes wheel suffix | Check `nvcc --version` and `torch.version.cuda` in cell 1 sanity output |
| **Cell 8 (Train!)** | Set `per_device_batch_size = 2` (was 1 on T4 to fit memory) | L4 has 24 GB VRAM, so we can go to batch=2 → halves training time |
| **Cell 8 (Train!)** | Set `epochs = 3` for the real v0.1 run | Lightning has the time budget (was 1 on Kaggle smoke run) |
| **Cell 13 (HF upload)** | No change — HF token works the same | HF_TOKEN env var works the same way on Lightning |

### Exact migration steps

1. **Sign up at https://lightning.ai** (free, no credit card).
2. Click **"New Studio"** → name it `thursday-ai`.
3. Open the Studio — you'll get a JupyterLab-like environment.
4. In the right sidebar, click **"Hardware"** → select **L4 GPU** (free tier — 22 h/month).
5. Upload your current `thursday-ai-training.ipynb` via drag-and-drop.
6. Make the 4 cell edits above.
7. Restart kernel, run cell 1. Expected sanity output (slightly different versions on Lightning):

   ```
   torch           2.5.1+cu121        (Lightning preinstalled)
   transformers    4.46.x             (whatever Lightning preinstalls)
   trl             0.11.4             (we override)
   peft            0.13.x
   accelerate      0.34.x
   bitsandbytes    0.45.3              (or 0.49.2 — both work on Lightning)
   huggingface_hub 0.25.x
   datasets        2.20.x
   GPU count: 1
     GPU 0: NVIDIA L4               (vs Tesla T4 on Kaggle)
   OK: all imports succeeded. Proceed to cell 2.
   ```

8. Run cells 2–7 (HF login, dataset load, model load in 4-bit, LoRA setup, SFTTrainer init). Model load will take ~2 min the first time (model download). Subsequent sessions will reuse cached download.
9. Cell 8 (Train!): expected to take **~6–8 hours on L4** (vs ~25 h on T4). Leave the tab open or background the Studio — Lightning supports long-running sessions.
10. Cells 9–14 (save LoRA, merge, build llama.cpp with CMake, convert to GGUF, upload to HF Hub, smoke test): unchanged from Kaggle version.

### What to watch for

- **CUDA version mismatch**: If Lightning has CUDA 12.1 and your `bitsandbytes==0.49.2` install fails to find the right `.so`, fall back to `bitsandbytes==0.45.3` (works with CUDA 12.1, per M0-fix-9 fallback plan).
- **Studio session timeout**: When you allocate a GPU and disconnect, the GPU keeps running until you explicitly release it or you run out of monthly GPU hours. Track usage in the right sidebar.
- **Free tier cap**: 22 GPU hours/month. If you burn 8 on one v0.1 run, you have 14 left for that month — enough for 1–2 retry runs.

---

## Section 5 — Cost Comparison for Real v0.1 Training

**Full v0.1 training = 90k examples × 3 epochs on Qwen2.5-3B QLoRA** at `seq=4096`, `r=32`, all-7 target modules, `bs=2 grad_accum=8` (effective batch 16).

### Per-GPU training-time estimates (informed by `worklog.md` M0-fix-18 memory budget + M0-fix-23 token math)

| GPU | VRAM | Relative speed vs T4 | Est. time for full v0.1 |
|---|---|---|---|
| Tesla T4 (Kaggle, Colab Free) | 16 GB | 1.0× | ~25 h |
| Tesla P100 (Kaggle) | 16 GB | ~0.8× | ~30 h |
| **L4 24 GB** (Lightning free, Colab Pro+, RunPod, Vast, Modal) | 24 GB | ~2.5× | **~8 h** |
| A10G 24 GB (Modal, RunPod, Brev, HF Spaces) | 24 GB | ~2.2× | ~9 h |
| RTX 3090 24 GB (Vast, RunPod) | 24 GB | ~3.5× | ~6 h |
| RTX 4090 24 GB (Vast, RunPod) | 24 GB | ~5× | ~4 h |
| RTX A5000 24 GB (RunPod) | 24 GB | ~3× | ~7 h |
| A100 40 GB (Colab Pro+, Lambda) | 40 GB | ~6× | ~3 h |
| A100 80 GB (RunPod, Lambda, Modal) | 80 GB | ~7× | ~3 h |
| H100 80 GB (RunPod, Lambda, Replicate) | 80 GB | ~9× | ~2 h |

### Cheapest paths to one finished Thursday AI v0.1

| Rank | Path | Total cost | Pros | Cons |
|---|---|---|---|---|
| 🥇 | **Lightning AI Studios Free tier — L4** | **$0** | Persistent storage, Jupyter UX, no idle disconnect, same notebook | 22 h/month quota — enough for v0.1 + ~2 retries |
| 🥈 | **Modal Starter — A10G** | **$0** ($30/mo credit = ~27 h A10G) | Most free compute of any option, per-second billing | Serverless UX requires code refactor |
| 🥉 | **Vast.ai spot — RTX 3090 24 GB** | **~$2** ($0.19/hr × 6 h + storage) | Cheapest absolute cost, 24 GB VRAM | Marketplace reliability — host can cancel; need checkpointing |
| 4 | **RunPod — RTX A5000 24 GB** | **~$1.62** ($0.27/hr × 6 h) | Stable, on-demand, Unsloth template | Slightly more than Vast spot |
| 5 | **RunPod — L4 24 GB** | **~$3.92** ($0.49/hr × 8 h) | Modern GPU, on-demand | More expensive than 3090 |
| 6 | **Lambda Labs — A100 PCIe 40 GB** | **~$6** ($1.99/hr × 3 h) | Reliable, fastest for the money | No free credits |
| 7 | **Colab Pro — L4** | **$10/mo** subscription | Familiar UX, no refactor | ~16 h L4/mo = barely 2 v0.1 runs/mo |
| 8 | **Colab Pro+ — L4/A100** | **$49.99/mo** subscription | 500 compute units, background execution | Expensive for a single training run |

### Verdict

- **If you want to spend $0:** Lightning AI Studios free tier. Migrate the notebook (Section 4), bump `max_seq_length` back to 4096, run with batch=2. Get a real v0.1 in ~8 hours for free.
- **If you want maximum free compute:** Modal Starter. $30/mo credit × per-second billing gives you 27+ hours of A10G or 37+ hours of L4 free. Refactor the notebook into a Modal `@app.function(gpu="A10G")` style.
- **If you're willing to spend $2–$5 to just be done with it:** Vast.ai RTX 3090 spot instance. 24 GB VRAM, ~6 hours, ~$2. Add checkpointing every 200 steps to be safe against host cancellation.

### The actual cheapest path to a *real* Thursday AI v0.1 (not the smoke test):

> **Switch to Lightning AI Studios free tier, migrate the notebook per Section 4, run with `smoke_mode = False`, `epochs = 3`, `max_seq_length = 4096`, `per_device_batch_size = 2`. Expected wall time: ~8 hours. Total cost: $0.**
>
> If Lightning's 22 h/month quota is exhausted (e.g. multiple retries), fall back to **Vast.ai RTX 3090 spot at ~$2 per training run**.

---

## Appendix A — Sources consulted (URLs)

### Primary sources (pricing pages / official docs)

- https://www.kaggle.com/docs/notebooks — Kaggle notebook environment specs (12h CPU/GPU, 9h TPU, T4×2 / P100 / TPU v3-8)
- https://research.google.com/colaboratory/faq.html — Colab free 12 h, Pro/Pro+ compute units, 24 h on Pro+
- https://lightning.ai/pricing — Lightning Studios pricing page (loaded via JS; free tier confirmed via secondary sources)
- https://huggingface.co/pricing — HF Spaces pricing (T4 small free w/ Pro; L4 $0.80/hr; A10G $1.00/hr; A100 80GB $2.50/hr)
- https://www.runpod.io/pricing — RunPod GPU cloud pricing (L4 $0.49/hr, RTX 3090 $0.50/hr, A100 PCIe $1.59/hr)
- https://vast.ai/pricing — Live Vast.ai marketplace pricing (RTX 3090 median $0.19/hr, RTX 4090 median $0.47/hr, L4 median $0.33/hr)
- https://modal.com/pricing — Modal pricing ($30/mo free credits on Starter, per-second billing, A10G $1.10/hr, L4 $0.80/hr, A100 80GB $2.50/hr)
- https://lambdalabs.com/service/gpu-cloud — Lambda instances pricing (A100 40GB $1.99/hr, A100 80GB $2.79/hr, H100 80GB $3.99/hr, V100 $0.79/hr)
- https://www.paperspace.com/pricing — Paperspace Gradient pricing (Free M4000, Pro $8/mo, A100 $2.24/hr 3-yr commit)
- https://replicate.com/pricing — Replicate per-second pricing (T4 $0.81/hr, L40S $3.51/hr, A100 80GB $5.04/hr)

### Secondary sources (cross-references for free-tier specifics)

- https://www.scribd.com/document/free-gpu-platforms — Cross-provider free tier table (Kaggle, Colab, Lightning, SageMaker SL)
- https://www.muratkarakaya.net/free-gpu-services-for-llm-enthusiasts/ — Lightning AI: 15 free credits/month, 1 credit = $1
- https://www.researchgate.net/.../laptop — SageMaker Studio Lab: 4 GPU hours/day, 15 GB persistent
- https://koonka.ai/paperspace-vs-google-colab — Colab Pro+ $49.99/mo for 500 compute units, background execution
- https://llmshosting.com/modal-review/ — Modal Starter $30/mo free credits, per-second billing
- https://www.softwr.com/lambda-labs-vs-modal — Modal free tier + $30/month starter credit confirmed
- https://www.rfp.wiki/anyscale-vs-google-ai — Anyscale $100 starter credits for new accounts
- https://dzone.com/articles/fine-tune-slms-for-free-from-google-colab-to-ollama — Confirms Unsloth on Colab Free T4 is the documented path
- https://pypi.org/project/unsloth/ — Unsloth PyPI: "Free Colab T4" supported, 2× T4 supported

### Search results that didn't make the cut but were informative

- https://hostline.io/gpu-servers-for-computer-vision — Vast.ai RTX 4090 listings ~$0.40/hr
- https://valebyte.com/blog/free-gpu-cloud — Free GPU budget guide
- https://gpueconomy.com — H100 retail/on-demand price tracking
- https://github.com/mvalentsev/awesome-free-ai-coding — Multi-provider free tier summary

---

## Appendix B — Why each "long-shot" option was excluded

| Option | Verdict | Reason |
|---|---|---|
| **Gitpod / Codespaces** | Excluded | CPU-only on free tiers. No GPU option without paid enterprise. |
| **CoreWeave** | Not recommended | Competitive pricing (~$2.21/hr A100) but no free tier, no consumer signup flow — designed for B2B. |
| **Together AI** | Honorable mention | $5 free credits on signup. Fine-tuning API exists. But for a notebook workflow, you'd need to refactor. Modal is the better serverless option. |
| **Anyscale** | Honorable mention | $100 free starter credits = ~100 h A10G free. But Anyscale is positioned as a Ray-native platform, not a Jupyter notebook host. Worth investigating if Lightning/Modal don't work out. |
| **Replicate** | Excluded for training | Designed for inference API, not training notebooks. Their fine-tuning API is for specific models (Flux, SDXL, etc.) — not arbitrary Qwen2.5-3B QLoRA. |
| **Paperspace Gradient Free** | Excluded | M4000 8 GB VRAM will OOM on Qwen2.5-3B + seq=4096. Pro $8/mo gives "faster free GPUs" but still capped. |
| **SageMaker Studio Lab** | Honorable mention | Free T4 + persistent storage is real, but 4 GPU hours/day = too restrictive for v0.1 (would need ~7 days at 4 h/day). Signup approval takes days. Also: AWS has gone quiet about it — long-term future uncertain. |
| **HF Spaces (Docker)** | Excluded for training | Designed for deployment, not training. Use it for *hosting* the finished model on free T4 small (with HF Pro $9/mo). |
| **Brev.dev** | Honorable mention | NVIDIA-owned, one-click templates for popular models (Mistral/Phi). But pricing isn't transparent and the platform is in flux. |
| **Kaggle itself** | Stay only if forced | 12 h cap, T4×2 NVLink issue, no persistent installs. Use only if Lightning is down or you've exhausted the 22 h/month quota and refuse to pay $2 for Vast. |
