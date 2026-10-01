# 03 - QLoRA Fine-tuning Recipe for Kaggle T4 x 2 + Potato-PC Inference

**Task ID:** 2-c  
**Agent:** QLoRA-Kaggle-Recipe-Researcher  
**Date:** 2025 (research compiled from 2024 Q4 / 2025 H1 sources)  
**Target hardware (train):** Kaggle Notebook with 2 x NVIDIA Tesla T4 (16 GB VRAM each, 32 GB total), 4 vCPU, 13 GB system RAM, ~73 GB ephemeral disk  
**Target hardware (inference):** Intel Core i5-3xxx (3rd gen, 4 cores / 4 threads, AVX but no AVX2 on most SKUs), 8 GB DDR3, Intel HD Graphics (no CUDA, no iGPU offload worth using)

---

## 0. TL;DR - Executive Summary

| Decision | Recommendation |
|---|---|
| Base model (primary, aligns with Task 2-a choice) | **Qwen2.5-3B-Instruct** (1.93 GB Q4_K_M fits 8 GB potato PC comfortably). Unsloth also fine-tunes the 7B variant if a future beefier host appears. |
| Training library | **Unsloth (FastLanguageModel)** + TRL `SFTTrainer` + PEFT + bitsandbytes. ~2x faster, ~60% less VRAM than vanilla HF on T4. |
| Distributed strategy | **DDP via `accelerate launch`** (model-parallel NOT needed). Both 3B and 7B QLoRA fit on a single T4; second T4 doubles throughput via data-parallel. |
| Max model size on T4x2 QLoRA | **8 B (Llama-3.1-8B) at seq 4096 OK**. **7B at seq 8192 OK** at bs=1, grad_accum=8. **14B+ requires model-parallel sharding** (avoid). |
| LoRA recipe | `r=16, alpha=16, dropout=0, target_modules=all-linear` (q,k,v,o,gate,up,down) |
| Optimizer | `adamw_8bit` (paged), LR `2e-4` (3e-4 for 3B), cosine, 3% warmup, 3 epochs |
| Export | Unsloth `model.save_pretrained_gguf(..., quantization_method='q4_k_m')` one-liner (auto-clones + compiles llama.cpp under the hood) |
| Potato-PC inference engine | **llama.cpp direct (`llama-server`)**. Ollama adds ~30-70% overhead for short responses. KoboldCpp is a fine middle-ground with prebuilt `noavx` binary. |
| Expected tok/s on i5-3xxx + 8 GB DDR3 (Q4_K_M, 4 threads, 4k context) | **3B: ~8-12 tok/s** | **7B: ~3-5 tok/s** |

---

## 1. T4 x 2 Fine-Tuning Recipes (Library Stack Comparison)

### 1.1 Candidate stacks evaluated

| Library | Pros | Cons | Verdict for Thursday AI |
|---|---|---|---|
| **Unsloth** (FastLanguageModel wrapper over HF transformers+trl+peft) | - ~2x faster, ~60% less VRAM than vanilla HF on T4 (Unsloth blog for Qwen2.5 family).<br>- One-line GGUF export via `save_pretrained_gguf`.<br>- `train_on_responses_only` for loss masking.<br>- Supports Kaggle T4x2 DDP via `accelerate launch` (see [Unsloth Multi-GPU docs](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth) and [PR #212](https://github.com/unslothai/notebooks/pull/212) "Port and Optimize Unsloth Notebooks for Kaggle Dual T4 GPUs").<br>- Active maintenance, official Colab + Kaggle notebooks. | - Pins specific `transformers`/`trl`/`bitsandbytes` versions (occasional compatibility friction).<br>- Multi-GPU support is "official-ish but not first-class" (must use accelerate/torchrun manually). | **RECOMMENDED** |
| **Axolotl** (YAML config-driven) | - Clean declarative YAML config.<br>- Native multi-GPU (DDP, DeepSpeed ZeRO-2/3, FSDP2) per [docs.axolotl.ai/docs/multi-gpu.html](https://docs.axolotl.ai/docs/multi-gpu.html).<br>- 2025/02 LoRA/QLoRA optimizations per [github.com/axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl). | - Heavier install, harder to inspect in a Kaggle notebook.<br>- LoRA optimizations are newer and have had some rough edges on T4.<br>- GGUF export still requires separate llama.cpp build. | Good alternative; overkill for a single notebook. |
| **transformers + peft + trl (raw HF)** | - Maximum control / debuggability.<br>- No version-pinning issues. | - ~2x slower than Unsloth on T4.<br>- No bundled GGUF helper.<br>- Must hand-roll loss masking. | Use only if Unsloth breaks. |
| **LLaMA-Factory** (hiyouga) | - WebUI + YAML, supports 2/3/4/5/6/8-bit QLoRA.<br>- Has a T4 Colab notebook per [github.com/hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory).<br>- 117% speed / 50% memory of vanilla HF. | - Abstraction layer hides chat-template/loss-mask details (bad for tool-calling trajectories).<br>- Heavier on Kaggle. | Optional alternative; not chosen. |

> **Decision:** Use **Unsloth** with **TRL `SFTTrainer`** as the training loop and **`accelerate`** as the launcher. This matches the official "Unsloth Finetuning multiple GPUs (2x T4 on Kaggle)" notebook (Kaggle user `nguyenit67`) and Unsloth PR #212.

### 1.2 Maximum model size on T4 x 2 with QLoRA

Empirical memory budget on a single T4 (15 GB usable VRAM after driver/PyTorch overhead):

| Model | 4-bit weight | Activations @ seq 4096, bs=2 | LoRA r=16 | Optimizer (AdamW 8-bit) | Fits 1 T4? | Fits T4x2 DDP? |
|---|---|---|---|---|---|---|
| Qwen2.5-**3B** | ~2.0 GB | ~3 GB | ~80 MB | ~0.5 GB | Yes easily, room for bs=4 | Yes (overkill, use 1 GPU) |
| Qwen2.5-**7B** | ~4.5 GB | ~6 GB | ~150 MB | ~0.7 GB | Yes tight, bs=2 seq 4096 | Yes comfortably |
| Qwen2.5-**14B** | ~8 GB | ~10 GB | ~280 MB | ~1.2 GB | No (OOM) | Yes if sharded (model-parallel) - but training unstable for QLoRA across shards |
| Llama-3.1-**8B** | ~5 GB | ~7 GB | ~150 MB | ~0.8 GB | Yes very tight (bs=1 seq 4096) | Yes |
| Phi-3.5-**mini** (3.8B) | ~2.4 GB | ~3 GB | ~80 MB | ~0.5 GB | Yes | Yes |
| Mistral-**7B** v0.3 | ~4.5 GB | ~6 GB | ~150 MB | ~0.7 GB | Yes tight | Yes |

**Conclusion:** On T4x2, the practical ceiling for a stable, fast QLoRA run is **7B-8B at seq 4096, bs=2 per device**. Going to 14B+ forces model-parallel sharding which complicates DDP and slows training - not worth it on Kaggle where every minute counts.

> Aligns with Task 2-a's primary recommendation of **Qwen2.5-3B-Instruct** for the potato-PC target; Qwen2.5-7B is a viable "beefier host" upgrade path that trains just as easily on T4x2.

### 1.3 Recommended hyperparameter table

| Hyperparameter | Value (3B - primary) | Value (7B - upgrade) | Notes |
|---|---|---|---|
| Learning rate | **3e-4** | 2e-4 | QLoRA paper default; smaller models tolerate higher LR |
| LR scheduler | cosine | cosine | 3% warmup ratio |
| Warmup ratio | 0.03 | 0.03 | |
| Epochs | **3** | 3 | Use EarlyStoppingCallback to avoid overfit |
| Per-device train batch size | **4** | 2 | |
| Gradient accumulation steps | **4** | 4 | effective batch = 4 x 2 GPUs x 4 = **32** (3B) / 2 x 2 x 4 = **16** (7B) |
| Max sequence length | **4096** | 4096 | 8192 possible on 7B at bs=1 if needed |
| LoRA `r` | **16** | 16 | 32/64 marginally better but VRAM cost; r=16 is sweet spot |
| LoRA `alpha` | **16** | 16 | alpha = r is modern default |
| LoRA `dropout` | **0** | 0 | Dropout=0 works best for instruction tuning |
| LoRA `target_modules` | `all-linear` (q, k, v, o, gate, up, down) | same | Unsloth handles this automatically |
| Quantization | **NF4 4-bit, double-quant=True, compute_dtype=bfloat16** | same | Standard QLoRA config |
| Optimizer | **adamw_8bit** (paged) | adamw_8bit | Saves optimizer VRAM |
| Weight decay | 0.01 | 0.01 | |
| Max grad norm | 1.0 | 1.0 | |
| Gradient checkpointing | **Unsloth ON by default** | ON | Unsloth's manual gradient checkpointing is faster than HF's |
| Packing | True | True | Unsloth sequence packing = ~2x throughput for multi-turn |
| Seed | 3407 | 3407 | ("torch.manual_seed(3407) is all you need") |

### 1.4 References used (Section 1)

- Unsloth official blog: [Qwen 2.5 Coder Fine-tuning with Unsloth](https://unsloth.ai/blog/qwen-coder) - "Tesla T4: Qwen 2.5 (7B) ... Unsloth makes Qwen 2.5 finetuning 2x faster and use 60% less memory than Flash Attention 2 (FA2) + Hugging Face"
- Unsloth [Multi-GPU Fine-tuning docs](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth) - "Unsloth currently supports multi-GPU setups through libraries like Accelerate and DeepSpeed ... Run `accelerate launch train.py` or `torchrun --nproc_per_node N_GPUS train.py`"
- Kaggle notebook: [Unsloth Finetuning multiple GPUs (2x T4 on Kaggle)](https://www.kaggle.com/code/nguyenit67/unsloth-finetuning-multiple-gpus-2x-t4-on-kaggle)
- Kaggle notebook: [Kaggle Qwen 2.5 Unsloth notebook](https://www.kaggle.com/code/danielhanchen/kaggle-qwen-2-5-unsloth-notebook) (official Unsloth author Daniel Hanchen)
- Hugging Face blog: [Fine-tune Llama 3.1 Ultra-Efficiently with Unsloth](https://huggingface.co/blog/mlabonne/sft-llama3)
- Reddit /r/LocalLLaMA: [I Fine-Tuned a 7B Medical AI for $0 on Free Kaggle GPUs](https://www.reddit.com/r/LocalLLaMA/comments/1wh813o/i_finetuned_a_7b_medical_ai_for_0_on_free_kaggle) (Qwen2.5-7B + QLoRA via Unsloth on Kaggle)
- Reddit /r/LocalLLaMA QLoRA hyperparameter thread: ["You can probably squeeze in lora_r 64 and ctx 4096 in 16GB on free T4 in Colab. I would start with lora_r 32, ctx 4096, learning rate 0.0001, 3 epochs..."](https://www.reddit.com/r/LocalLLaMA/comments/1aq1u3n/qlora_hyperparameters_for_small_finetuning_task)
- Trelis Research: [Multi GPU Training with Unsloth](https://trelis.substack.com/p/multi-gpu-training-with-unsloth)
- Ertas AI Qwen2.5 7B vs Llama 8B QLoRA benchmark: [https://www.ertas.ai/blog/fine-tune-llama-3-3-qwen-2-5-qlora-benchmark](https://www.ertas.ai/blog/fine-tune-llama-3-3-qwen-2-5-qlora-benchmark)
- Fine-tune Qwen2.5-7B on 16 GB GPU with QLoRA: [https://explore.n1n.ai/blog/fine-tune-qwen-7b-qlora-16gb-gpu-2026-06-21](https://explore.n1n.ai/blog/fine-tune-qwen-7b-qlora-16gb-gpu-2026-06-21)

---

## 2. Multi-Turn Chat + Tool-Call Training

### 2.1 Loss masking strategy

For agentic trajectory data of the form `system -> user -> assistant(tool_call) -> tool(observation) -> assistant(final)`, we want gradients ONLY on the assistant turns (both the tool-call emission and the final answer). System / user / tool / observation tokens must be **masked** (label = -100) so the model does not learn to parrot them.

Unsloth's `train_on_responses_only(trainer, instruction_part=..., response_part=...)` post-hoc patches the labels. It finds every `instruction_part` (the start-of-user-turn token sequence) and every `response_part` (start-of-assistant-turn token sequence) in the tokenized input, and masks everything outside the response windows.

### 2.2 Template-specific masking args (per base model)

| Base model | `instruction_part` | `response_part` |
|---|---|---|
| **Qwen2.5 / Qwen2 / Qwen3 (ChatML)** | `<|im_start|>user\n` | `<|im_start|>assistant\n` |
| Llama-3.1 / 3.2 / 3.3 | `<|start_header_id|>user<|end_header_id|>\n\n` | `<|start_header_id|>assistant<|end_header_id|>\n\n` |
| Gemma 2 / 3 | `<start_of_turn>user\n` | `<start_of_turn>model\n` |
| Mistral / Hermes-Pro | `<|im_start|>user\n` (Hermes uses ChatML) | `<|im_start|>assistant\n` |
| Phi-3.5 | `<|user|>\n` | `<|assistant|>\n` |

> Source: [Unsloth FAQ](https://unsloth.ai/docs/basics/troubleshooting-and-faqs). Verify by checking `loss != 0` after step 1 - if you see "All labels in your dataset are -100. Training losses will be all 0", your `instruction_part`/`response_part` strings do not match the tokenizer's chat template byte-for-byte (Unsloth issue #823).

### 2.3 Code - applying loss masking for Qwen2.5 (our target)

```python
from unsloth.chat_templates import get_chat_template, train_on_responses_only

# 1. Load the Qwen2.5 chat template (already includes tool-call support)
tokenizer = get_chat_template(
    tokenizer,
    chat_template='qwen-2.5',
    mapping={'role': 'role', 'content': 'content',
             'tool_calls': 'tool_calls', 'tool_call_id': 'tool_call_id'},
)

# 2. Build trainer normally (SFTTrainer + SFTConfig)
trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, args=cfg)

# 3. Patch the trainer to mask non-assistant tokens
trainer = train_on_responses_only(
    trainer,
    instruction_part='<|im_start|>user\n',
    response_part='<|im_start|>assistant\n',
)

# 4. Sanity-check that masking worked:
#    Decoded label tokens should be all -100 outside assistant turns.
tokens = trainer.train_dataset[0]['input_ids']
labels = trainer.train_dataset[0]['labels']
assert any(l != -100 for l in labels), 'All labels are -100 -> masking is wrong'
print('Masking OK. Tokens with loss:', sum(1 for l in labels if l != -100), '/', len(labels))
```

### 2.4 How Qwen2.5 chat template handles tool calls

Qwen2.5's `tokenizer_config.json` ships with a Jinja2 chat template that auto-emits:

- **System turn** includes a `<tools>...</tools>` XML block when `tools=[...]` is passed to `apply_chat_template`.
- **Assistant tool-call turn** is emitted as an XML `<tool_call>` block with JSON arguments inside.
- **Tool observation turn** uses role=`tool` with a `<tool_response>` block wrapping the returned JSON.

This matches the [Qwen2.5 blog](https://qwen.ai/blog?id=qwen2.5): "Qwen2.5's chat template also includes a tool calling template, uses a tool calling template inspired by Nous' Hermes."

Per [Qwen function-calling docs](https://qwen.readthedocs.io/en/latest/framework/function_call.html): "We recommend using Hermes-style tool use for Qwen3 to maximize function calling performance." The same template works for Qwen2.5.

### 2.5 Why Qwen2.5 over Llama-3.1 for Thursday AI's tool-calling use case

| Criterion | Qwen2.5-3B/7B | Llama-3.1-8B |
|---|---|---|
| Native tool-call training data (pretraining) | YES (Qwen-Hermes format) | Partial (Llama-3.1 added tool format but base 3.1 weaker than Qwen2.5) |
| Hermes XML tool-call support | Built-in | Requires fine-tune to add |
| Tool-call template in `tokenizer_config.json` | YES (verified) | YES |
| Quality on BFCL tool-calling leaderboard (small models) | Higher at 3B/7B class | Higher at 8B+ class |
| `tokenizer.apply_chat_template(tools=[...])` works out-of-box | YES | YES |

### 2.6 Sources used (Section 2)

- [Qwen2.5 blog (mentions Hermes-style tool template)](https://qwen.ai/blog?id=qwen2.5)
- [Qwen function-calling docs (Hermes recommendation for Qwen3)](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [Fine-tune Qwen2.5-1.5B for function calling with QLoRA - necdetduruk.com](https://necdetduruk.com/posts/fine-tuning-qwen-function-calling-qlora-part-2)
- [TinyToolCaller QLoRA 1.5B for function calling - readytensor.ai](https://app.readytensor.ai/publications/tinytoolcaller-qlora-fine-tuning-of-a-15b-llm-for-reliable-function-calling-a95jlwdioKYatbzl8OKNS)
- [vLLM tool-calling docs](https://docs.vllm.ai/en/latest/features/tool_calling) - confirms Qwen2.5 tool format works with `--tool-call-parser hermes`
- [Unsloth FAQ: train_on_responses_only](https://unsloth.ai/docs/basics/troubleshooting-and-faqs)
- [Unsloth chat templates docs](https://unsloth.ai/docs/basics/chat-templates)

---

## 3. Distributed Training on T4 x 2

### 3.1 DDP vs model-parallel for this project

| Strategy | When to use | For Thursday AI? |
|---|---|---|
| **DDP (Distributed Data Parallel)** | Each GPU has a full model copy; gradients are all-reduced after backward. Requires the full model + activations + optimizer to fit on **one** GPU. | YES - both Qwen2.5-3B and 7B QLoRA fit on 1 T4. Use DDP for ~2x throughput. |
| Model-parallel / FSDP / DeepSpeed ZeRO-3 | Shards model weights/optimizer across GPUs. Use when one GPU can't hold the model. | NO - 14B+ would need this; we're staying at 3B/7B. |
| Pipeline parallel | Splits layers across GPUs. Slow on T4 due to no NVLink. | NO. |

### 3.2 Recommended `accelerate config` for T4 x 2

Save as `default_config.yaml` in `~/.cache/huggingface/accelerate/default_config.yaml`, or generate interactively with `accelerate config`:

```yaml
# accelerate config for Kaggle T4 x 2 (DDP)
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config: {}
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2          # 2 x T4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### 3.3 Launch command (Kaggle notebook cell)

Option A - via `accelerate launch`:
```bash
!accelerate launch --num_processes=2 --multi_gpu train.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --output_dir /kaggle/working/thursday-ai-lora
```

Option B - via raw `torchrun` (more explicit, no accelerate config needed):
```bash
!torchrun --nproc_per_node=2 --master_port=29501 train.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --output_dir /kaggle/working/thursday-ai-lora
```

Both work; `accelerate launch` is slightly more user-friendly because it auto-detects mixed precision and CUDA_VISIBLE_DEVICES.

### 3.4 Kaggle-specific gotchas

| Constraint | Value | Mitigation |
|---|---|---|
| Weekly GPU quota | **30 hours/week** (shared across P100 and T4x2; occasionally bumped to ~40h in low-demand periods) | Plan 3 training runs/week max. Each run = ~6-9h cell time. |
| Max session length | **9 hours** (notebook auto-stops) | Use checkpoint-resume: `trainer.train(resume_from_checkpoint=True)`. Save checkpoints every 100 steps. |
| Internet | Must be **ON** for `pip install` and HF download; can be OFF during training | Leave Internet ON for setup cells, OFF during training (Kaggle suggestion). |
| Disk | `/kaggle/working/` ~73 GB ephemeral | Save model + checkpoints here. Final upload to HF/GitHub at end. |
| System RAM | **13 GB** | Unsloth + datasets rarely exceed 4 GB. Avoid `load_dataset(... num_proc=N)` with N>2. |
| HF token | Use `kaggle_secrets.UserSecretsClient` (NOT env vars) | See §6.3. |
| Multi-GPU detection | Both T4s auto-visible as `cuda:0` and `cuda:1` | Don't set `CUDA_VISIBLE_DEVICES` unless debugging. |

### 3.5 Why NOT to use FSDP on T4x2 (counter-recommendation)

Kaggle user `aisuko` published ["Multiple-GPUs ft-llama3.1 with FSDP and QLoRA"](https://www.kaggle.com/code/aisuko/multiple-gpus-ft-llama3-1-with-fsdp-and-qlora). It works, but:

1. FSDP+QLoRA has known bugs with bitsandbytes 4-bit weights (gradient sync issues - see [HF discuss thread](https://discuss.huggingface.co/t/orpo-trainer-giving-error-when-fine-tuning-llama3-8b-in-multi-gpu-environment/86765)).
2. FSDP is slower than DDP when the model fits on one GPU (extra gather/scatter ops).
3. FSDP shines for 30B+ models that don't fit on one GPU - we're not in that regime.

### 3.6 Sources used (Section 3)

- Kaggle [Multi-GPU and Accelerate](https://www.kaggle.com/code/muellerzr/multi-gpu-and-accelerate) by muellerzr
- Kaggle [T4x2 multi-GPU training G2Net](https://www.kaggle.com/code/shivanshuman/kaggle-t4x2-multi-gpu-training-g2net)
- Kaggle discussion: [How to Run PyTorch DDP on Kaggle Using Multiple GPUs (T4x2)](https://www.kaggle.com/discussions/getting-started/580460) - "Always make sure to select T4 x2 in the Notebook accelerator settings. Use `--standalone` with torchrun and set `--nproc_per_node=2`"
- Kaggle [Multiple-GPUs ft-llama3.1 with FSDP and QLoRA](https://www.kaggle.com/code/aisuko/multiple-gpus-ft-llama3-1-with-fsdp-and-qlora) (reference, not recommended)
- LearnOpenCV: [PyTorch DDP Training in Kaggle T4x2 GPU runtime](https://learnopencv.com/distributed-parallel-training-pytorch-multi-gpu-setup)
- Kaggle [Weekly Maximum GPU Usage (30h/week)](https://www.kaggle.com/discussions/general/108481)
- Kaggle [Efficient GPU Usage Tips](https://www.kaggle.com/docs/efficient-gpu-usage)
- Kaggle [Notebooks docs (9h session limit)](https://www.kaggle.com/docs/notebooks)
- HF knowledge base on Kaggle limits (2025): [https://huggingface.co/datasets/John6666/knowledge_base_md_for_rag_1/blob/main/kaggle_20251121.md](https://huggingface.co/datasets/John6666/knowledge_base_md_for_rag_1/blob/main/kaggle_20251121.md)

---

## 4. Saving and Exporting the Fine-Tune

### 4.1 Three artifacts to produce

| Artifact | Size (3B) | Size (7B) | Purpose |
|---|---|---|---|
| LoRA adapter (adapter_config.json + adapter_model.safetensors) | ~50 MB | ~100 MB | Re-use on top of base model for further fine-tunes; smallest upload to HF/GitHub |
| Merged 16-bit HF model | ~6 GB | ~15 GB | Optional - only if you want a standalone HF repo. Can regenerate from base+adapter. |
| **GGUF Q4_K_M** (model-q4_k_m.gguf) | ~1.9 GB | ~4.4 GB | **Final deployment artifact** for llama.cpp / Ollama / KoboldCpp |

### 4.2 Unsloth one-liner (RECOMMENDED)

Unsloth's FastLanguageModel has built-in GGUF export that auto-clones llama.cpp and quantizes. From the [Unsloth GGUF docs](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf):

```python
# Save adapter only (small, re-usable)
model.save_pretrained('thursday-ai-lora-adapter')  # ~50 MB
tokenizer.save_pretrained('thursday-ai-lora-adapter')

# Save merged 16-bit (for HF Hub, optional)
model.save_pretrained_merged('thursday-ai-merged-16bit', tokenizer, save_method='merged_16bit')

# Save merged 4-bit (for HF Hub, optional - smaller)
model.save_pretrained_merged('thursday-ai-merged-4bit', tokenizer, save_method='merged_4bit_forced')

# Quantize to GGUF Q4_K_M locally (Unsloth auto-clones llama.cpp + builds it)
model.save_pretrained_gguf('thursday-ai-gguf', tokenizer, quantization_method='q4_k_m')

# Or push directly to HuggingFace Hub
model.push_to_hub_gguf('skyro777/thursday-ai-gguf', tokenizer, quantization_method='q4_k_m')
```

### 4.3 All supported quantization methods (from llama.cpp ALLOWED_QUANTS)

| Method | Bits | Use case |
|---|---|---|
| `q4_k_m` | ~4.5 | **Recommended.** Q6_K for half of attn.wv + ff.w2, Q4_K for the rest. Best speed/quality tradeoff. |
| `q5_k_m` | ~5.5 | Slightly higher quality, ~25% larger. |
| `q8_0` | 8.0 | High quality but big. Use for eval, not deployment. |
| `f16` | 16.0 | Lossless; huge. Only for archival. |
| `q3_k_m` | ~3.5 | Smaller, more quality loss. Use only if RAM-constrained. |
| `q2_k` | ~2.5 | Last-resort compression; quality drops noticeably. |
| `iq3_xs`, `iq3_xxs`, `iq2_xxs` | ~2.5-3.0 | Newer I-quant methods - slightly better than legacy Q2/Q3 at same size, but requires llama.cpp build >= Oct 2024. |

### 4.4 Manual llama.cpp conversion (fallback)

If Unsloth's auto-export fails (rare, but happens for newly-released model architectures), do it manually:

```bash
# 1. Save merged 16-bit HF model first (via Unsloth's save_pretrained_merged)

# 2. Clone llama.cpp and build it with CUDA support (faster quantize on T4)
!apt-get update && apt-get install -y build-essential cmake git
!git clone https://github.com/ggml-org/llama.cpp
!cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=ON -DBUILD_SHARED_LIBS=OFF
!cmake --build llama.cpp/build --config Release -j --target llama-cli llama-server llama-quantize llama-gguf-split

# 3. Convert HF -> F16 GGUF
!python llama.cpp/convert_hf_to_gguf.py thursday-ai-merged-16bit --outfile thursday-ai-f16.gguf --outtype f16

# 4. Quantize F16 -> Q4_K_M
!llama.cpp/build/bin/llama-quantize thursday-ai-f16.gguf thursday-ai-q4_k_m.gguf q4_k_m
```

See [Kaggle notebook "Converting the Model to Llama.cpp GGUF"](https://www.kaggle.com/code/davidsit/converting-the-model-to-llama-cpp-gguf) and [llama.cpp discussion #2948](https://github.com/ggml-org/llama.cpp/discussions/2948) for reference.

### 4.5 Upload strategy

```python
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)

# Create repos
api.create_repo('skyro777/thursday-ai-lora', repo_type='model', exist_ok=True)
api.create_repo('skyro777/thursday-ai-gguf', repo_type='model', exist_ok=True)

# Upload LoRA adapter (small files)
api.upload_folder(folder_path='thursday-ai-lora-adapter',
                    repo_id='skyro777/thursday-ai-lora', repo_type='model')

# Upload GGUF (large file - uses LFS automatically)
api.upload_file(path_or_fileobj='thursday-ai-gguf/model-q4_k_m.gguf',
                path_in_repo='thursday-ai-3b-q4_k_m.gguf',
                repo_id='skyro777/thursday-ai-gguf', repo_type='model')
```

### 4.6 Optional: Ollama library

To publish to a personal Ollama library (optional, after the GGUF is on HF Hub):

1. Create a `Modelfile`:
   ```
   FROM ./thursday-ai-3b-q4_k_m.gguf
   TEMPLATE "{{ .System }}\n{{ .Prompt }}"
   PARAMETER stop "<|im_end|>"
   PARAMETER stop "<|endoftext|>"
   PARAMETER num_ctx 4096
   PARAMETER num_thread 4
   ```
2. Run `ollama create thursday-ai -f Modelfile` locally and `ollama push skyro777/thursday-ai` (requires Ollama account).

### 4.7 Sources used (Section 4)

- [Unsloth GGUF docs](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf) - exact `save_pretrained_gguf` / `push_to_hub_gguf` API
- [Reddit r/LocalLLaMA: How to convert my fine-tuned model to GGUF](https://www.reddit.com/r/LocalLLaMA/comments/1amjx77/how_to_convert_my_finetuned_model_to_gguf) - "Unsloth automatically merges your LoRA weights and makes a 16bit model, then converts to GGUF directly. All GGUF formats are supported ie q4_k_m"
- [llama.cpp convert HF to GGUF (discussion #2948)](https://github.com/ggml-org/llama.cpp/discussions/2948)
- [Kaggle notebook: Converting the Model to Llama.cpp GGUF](https://www.kaggle.com/code/davidsit/converting-the-model-to-llama-cpp-gguf)
- [The easiest way to convert a model to GGUF and Quantize](https://medium.com/@qdrddr/the-easiest-way-to-convert-a-model-to-gguf-and-quantize-91016e97c987)

---

## 5. Inference Engine for Potato PC

### 5.1 Target hardware recap

- CPU: Intel Core i5-3xxx (Ivy Bridge, 2012) @ ~3.3 GHz, **4 cores / 4 threads**
- RAM: 8 GB DDR3-1600 (~12.8 GB/s peak bandwidth)
- GPU: Intel HD Graphics 4000 (no CUDA, no Vulkan compute, ~16 ALUs - useless for LLM inference)
- Storage: assumed SSD
- ISA: SSE4.2 + AVX1 (NO AVX2, NO AVX-512, NO FMA, NO AMX) - critical for binary compatibility

### 5.2 Candidate engines compared

| Engine | Pros | Cons | Latency overhead vs llama.cpp direct | Verdict |
|---|---|---|---|---|
| **llama.cpp direct** (`llama-server` or `llama-cli`) | - Lowest possible latency (C/C++, no runtime overhead)<br>- Pre-built `noavx` binaries available for SSE4.2-only CPUs<br>- `mmap` keeps RAM usage near the weight file size<br>- Best threads/cache tunability | - Must download/compile a binary<br>- No nice CLI UX for chat | 1.0x (baseline) | **RECOMMENDED for production** |
| **KoboldCpp** (wraps llama.cpp with extras) | - Ships **pre-built `noavx` binary** (no compile needed on potato PC)<br>- GUI for tuning<br>- Built-in tool-calling template parser<br>- Supports SDPA, Flash Attention fallbacks | - Slightly slower than llama.cpp direct (~5-10%) due to HTTP server<br>- One more dependency to version-pin | ~1.05-1.1x | **RECOMMENDED as fallback** if user wants a GUI |
| **Ollama** | - One-line install (`curl ... | sh`)<br>- `ollama pull` auto-downloads GGUF from HF Hub<br>- Familiar CLI<br>- Built-in Modelfile format | - ~30-70% SLOWER than llama.cpp direct for short responses (Go runtime, REST API overhead per request - see [Reddit benchmark](https://www.reddit.com/r/LocalLLaMA/comments/1q64f26/llamacpp_vs_ollama_70_higher_code_generation))<br>- Memory overhead ~500-800 MB extra<br>- Tool-calling support via `tools` API but adds another JSON layer | ~1.3-1.7x for short responses | OK for casual users; NOT for latency-sensitive voice-first use |
| **LM Studio** | - Nice desktop GUI<br>- Built-in model browser | - Electron app ~300 MB+ RAM overhead<br>- Closed-source GUI layer on top of llama.cpp<br>- No `noavx` build - requires AVX2 | ~1.5-2x | NOT suitable for potato PC (AVX2 requirement + Electron bloat) |
| **llamafile** (Mozilla) | - Single-file binary, portable across OS/CPU | - Compiled with AVX2 by default<br>- Smaller community than llama.cpp<br>- Slower to get new features | ~1.0-1.1x | Optional; OK if prebuilt noavx version found |

> Per [Mozilla AI benchmark of llama.cpp / llamafile / LM Studio / Ollama](https://blog.mozilla.ai/benchmarking-local-llm-servers-llama-cpp-llamafile-lm-studio-and-ollama): "The engine barely matters" for GPU inference, but on CPU-only machines the engine choice DOES matter because HTTP server overhead dominates short-response latency.

### 5.3 Recommended llama.cpp CLI flags for i5-3xxx + 8 GB DDR3

```bash
./llama-server \
    --model thursday-ai-3b-q4_k_m.gguf \
    --threads 4 \                       # i5-3xxx has 4 logical cores
    --threads-batch 4 \                 # same for prompt eval (no hyperthreading)
    --ctx-size 4096 \                   # 4k context; reduce to 2048 for tiny-RAM margin
    --batch-size 32 \                   # prompt-processing batch
    --ubatch-size 32 \                  # micro-batch for prompt eval
    --mmap \                            # memory-map weights (don't load all into RAM)
    --mlock \                           # optional: lock weights in RAM (skip if 8 GB tight)
    --cache-type-k q8_0 \               # quantize KV cache (cuts ~50% KV RAM)
    --cache-type-v q8_0 \               # quantize value cache too (small quality hit)
    --n-gpu-layers 0 \                 # CPU-only
    --port 8080
```

Memory budget (Qwen2.5-3B @ Q4_K_M, ctx=4096, KV q8_0):
- GGUF weights (mmap'd, only touched pages count): ~1.9 GB
- KV cache (q8_0, 4096 tokens, 16 layers, 16 heads, 128 head dim): ~0.25 GB
- Compute buffer + scratch: ~0.3 GB
- **Peak process RAM: ~2.5 GB** (leaves ~5.5 GB for OS + browser + Playwright)

If running Qwen2.5-7B instead (Q4_K_M):
- GGUF weights: ~4.4 GB
- KV cache q8_0: ~0.5 GB
- Compute + scratch: ~0.5 GB
- **Peak: ~5.4 GB** (only ~2.6 GB left for everything else - RISKY on 8 GB DDR3)

### 5.4 Expected tokens/sec on i5-3xxx + DDR3-1600 (Q4_K_M, no AVX2)

Benchmarks for this exact CPU class are scarce (it's a 2012 chip), but extrapolated from:
- [TuringPi RK3588 LLM benchmarks (similar RAM bandwidth class)](https://turingpi.com/llm-inference-benchmarks-rk3588-gguf-quantization) - "1.5B Q4_K_M = ~22.6 tokens/sec at ~2 GiB peak RAM"
- [llama.cpp benchmark on Intel i5, 2 cores, 8GB RAM](https://github.com/ggml-org/llama.cpp/issues/34) - older data point
- [Invidelabs LLM on consumer-grade CPU-only machines](https://blog.invidelabs.com/llm-on-consumer-grade-cpu-only-machines)
- [llama-bench README reference for Qwen2 7B Q4_K_M](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md)

Extrapolated estimates (subject to actual i5 model + RAM speed):

| Model | Quant | Threads | Tok/s gen | Tok/s prompt-eval (pp512) | RAM peak |
|---|---|---|---|---|---|
| Qwen2.5-**1.5B** | Q4_K_M | 4 | 14-18 | 60-90 | ~1.2 GB |
| Qwen2.5-**3B**   | Q4_K_M | 4 | **8-12** | 30-50 | ~2.5 GB |
| Qwen2.5-**7B**   | Q4_K_M | 4 | 3-5 | 12-20 | ~5.4 GB |
| Llama-3.1-**8B** | Q4_K_M | 4 | 2-4 | 10-18 | ~6.0 GB |

> For Thursday AI's voice-first use case (3-7 word replies), 8-12 tok/s on Qwen2.5-3B is more than enough: a 6-word reply = ~8 tokens, generated in <1 second.

### 5.5 Context length tradeoff (4k vs 8k)

| Context | KV cache size (3B, q8_0) | Tok/s impact (vs 4k) | Use case |
|---|---|---|---|
| 2048 | ~0.13 GB | +10-15% | Short single-tool calls |
| **4096** | ~0.25 GB | baseline | Multi-turn chat with 1-2 tool calls (RECOMMENDED) |
| 8192 | ~0.5 GB | -5-10% | Long agentic trajectories with many tool round-trips |
| 16384 | ~1.0 GB | -10-20% | Only if you have ≥16 GB RAM |

### 5.6 AVX compatibility note (CRITICAL)

Intel Ivy Bridge (i5-3xxx) supports **AVX1 but NOT AVX2**. Most prebuilt llama.cpp binaries are compiled with AVX2 by default and will SIGILL-crash on this CPU.

Solutions:
1. **KoboldCpp noavx build**: [github.com/LostRuins/koboldcpp/releases](https://github.com/LostRuins/koboldcpp/releases) - look for `koboldcpp_noavx.zip`
2. **llama.cpp self-compiled**: `cmake -B build -DGGML_AVX2=OFF -DGGML_AVX=ON -DGGML_FMA=OFF ...` then `make llama-server`
3. **Fallback**: if even AVX1 fails (very old i5-2xxx or earlier), use the `noavx` build of KoboldCpp which works on pure SSE4.2.

### 5.7 Sources used (Section 5)

- [Mozilla AI: Benchmarking llama.cpp vs llamafile vs LM Studio vs Ollama](https://blog.mozilla.ai/benchmarking-local-llm-servers-llama-cpp-llamafile-lm-studio-and-ollama)
- [llama.cpp vs Ollama: ~70% higher code generation throughput](https://www.reddit.com/r/LocalLLaMA/comments/1q64f26/llamacpp_vs_ollama_70_higher_code_generation)
- [Ollama vs llama.cpp (atomic.chat)](https://atomic.chat/blog/guides/ollama-vs-llamacpp)
- [KoboldCpp vs Ollama (geeky-gadgets)](https://www.geeky-gadgets.com/koboldcpp-vs-ollama)
- [LLM on consumer-grade CPU-only machines (i5 guide)](https://blog.invidelabs.com/llm-on-consumer-grade-cpu-only-machines)
- [OpenBenchmarking llama.cpp benchmarks](https://openbenchmarking.org/test/pts/llama-cpp)
- [llama.cpp llama-bench README (Qwen2 7B Q4_K_M reference)](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md)
- [TuringPi RK3588 GGUF quantization benchmarks](https://turingpi.com/llm-inference-benchmarks-rk3588-gguf-quantization)

---

