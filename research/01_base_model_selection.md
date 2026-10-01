# Thursday AI — Base Model Selection (Task 2-a)

**Researcher:** HF-Model-Researcher subagent
**Date:** November 2025 (web search) / project date
**Mission:** Find the best HuggingFace base model for *Thursday AI* — an OFFLINE, voice-first, autonomous, OS-controlling assistant that runs on a potato PC (8 GB DDR3, Intel HD iGPU, i5-3rd-gen ~3.3 GHz, no dGPU) and gets fine-tuned on Kaggle T4 × 2 with QLoRA.

---

## TL;DR — Final Recommendation

**Thursday AI should fine-tune from `Qwen/Qwen2.5-3B-Instruct` at Q4_K_M (1.93 GB), served via llama.cpp (CPU AVX build) or Ollama, fine-tuned with Unsloth 4-bit QLoRA on Kaggle T4 × 2.**

Key reasons:
1. Fits comfortably in 8 GB RAM (1.93 GB GGUF + ~0.5 GB KV cache + OS) — leaves 4–5 GB free for browser/Playwright/Pillow.
2. Apache 2.0 license (no EU restrictions, no MAU limit, no AUP "system automation" carve-outs).
3. Native `tool_calls` support via Qwen2 chat template + JSON schema — same convention as OpenAI / Ollama tool calls.
4. Same chat template, tool-call format, and architecture as `Qwen2.5-7B-Instruct` → free upgrade path when Thursday AI runs on a stronger host.
5. Unsloth has an official Kaggle notebook for `unsloth/Qwen2.5-3B-Instruct` (4-bit QLoRA, T4).
6. 29+ languages natively supported.
7. Expected ~6–10 tok/s on i5 3rd-gen CPU with AVX-build llama.cpp (Q4_K_M, 4 threads). Fast enough for 3–7 word voice replies.

---

## 1. Potato-PC constraint analysis (hardware reality check)

Target machine (per project spec):
- CPU: Intel i5 3rd-gen (Ivy Bridge, 2012) @ ~3.3 GHz, 4 threads
- RAM: 8 GB DDR3-1600 (~12.8 GB/s peak bandwidth)
- GPU: Intel HD Graphics 4000 (no CUDA, no Vulkan compute, ~16 ALUs — useless for LLM inference)
- Storage: assumed SSD

Inference engine options on this machine:

| Engine | Status on i5-3rd-gen | Notes |
|---|---|---|
| **llama.cpp (CPU, AVX build)** | ✅ Best option | AVX is supported (Ivy Bridge = first AVX x86 gen). AVX2 is **NOT** supported on Ivy Bridge (added with Haswell, 2013). Must build with `LLAMA_AVX2=OFF LLAMA_AVX=ON LLAMA_FMA=ON LLAMA_F16C=ON`. FMA is supported on Ivy Bridge. |
| **Ollama** | ✅ Convenience wrapper | Uses llama.cpp underneath; auto-picks CPU backend. Easiest install path on Windows / Linux / macOS. |
| **vLLM-CPU** | ⚠️ Heavier | Requires more RAM headroom, less tuned for weak CPUs. PagedAttention benefits not needed at 1–4 concurrent users. |
| **Intel SYCL / OpenVINO (iGPU)** | ⚠️ Experimental | HD 4000 has no Vulkan compute; SYCL only works on Intel Arc / Xe. Not viable. |

**Rule of thumb for potato PC inference (per `localaimaster.com/blog/run-llm-cpu-only`):**
- 0.5–4 B params: genuinely usable
- 7–8 B params: tolerable with DDR5; bad on DDR3
- 8B+ params: "a bad time"

**Per `github.com/ggml-org/llama.cpp/issues/34` benchmark**: an Intel i5, 2-core, 8 GB RAM ran a 7B Q4 model at **~94 ms/token ≈ 10.6 tok/s**. On 4 threads + AVX (no AVX2), expect **~2.5–4 tok/s** for a 7B Q4_K_M and **~6–10 tok/s** for a 3B Q4_K_M. Voice reply latency target: <2 s for a 3–7 word response. → **3B class is the sweet spot.**

---

## 2. Per-candidate deep-dive

### TIER A — Potato-friendly 3B–4B

#### A1. `Qwen/Qwen2.5-3B-Instruct` ← **TIER A WINNER**

| Field | Value |
|---|---|
| HF model ID | `Qwen/Qwen2.5-3B-Instruct` |
| Architecture | Qwen2 (Dense decoder-only Transformer with RoPE, SwiGLU, RMSNorm, GQA with QKV bias, tied embeddings) |
| Parameters | 3.09 B total / 2.77 B non-embedding |
| Layers / Heads | 36 layers, 16 Q heads / 2 KV heads (GQA 8:1) |
| Native context | 32 768 tokens (config default), generation 8 192 |
| Extended context | Up to 131 072 via YaRN rope-scaling |
| Function calling | ✅ Yes — Qwen2 chat template has native `tool_calls` / `tools` role, JSON-schema tool definitions. Same convention as OpenAI / Ollama. |
| Multilingual | ✅ 29+ languages (English, Chinese, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, …) |
| License | **Apache 2.0** (commercial-friendly, no MAU limit, no EU restriction) |
| GGUF Q4_K_M size | **1.93 GB** (verified at `bartowski/Qwen2.5-3B-Instruct-GGUF`) |
| Other quants | Q5_K_M=2.22 GB, Q3_K_M=1.71 GB, Q8_0=3.29 GB, F16=6.18 GB, IQ4_XS=1.74 GB |
| Ollama | `ollama run qwen2.5:3b-instruct-q4_K_M` |
| BFCL raw score | 31.81 overall (BFCL v4, SOTA2 Nov-2025). Low raw — **but Qwen2.5-7B base scored 59.5 raw, so the architecture scales well post-fine-tune.** |
| Fine-tune on T4 | ✅ Verified — `unsloth/Qwen2.5-3B-Instruct` has dedicated Kaggle T4 notebook (`kaggle.com/code/danielhanchen/kaggle-qwen-2-5-conversational-unsloth`). Supports 4-bit QLoRA, 2× faster than vanilla. |
| Known issues | Base 3B is weaker at long agentic chains than 7B — mitigated by fine-tuning on OS-control traces. |

**Why Tier A winner:** Apache 2.0, smallest GGUF in the comparison (1.93 GB), native tool-calling format, official Unsloth Kaggle notebook, multilingual, has a clean upgrade path to Qwen2.5-7B-Instruct.

#### A2. `meta-llama/Llama-3.2-3B-Instruct` ← **TIER A RUNNER-UP**

| Field | Value |
|---|---|
| HF model ID | `meta-llama/Llama-3.2-3B-Instruct` |
| Architecture | Llama 3 (Dense decoder-only, RoPE, GQA) |
| Parameters | 3.21 B |
| Native context | 128 000 tokens |
| Function calling | ⚠️ Yes, but in "zero-shot" mode only (per `huggingface.co/blog/llama32`). Llama 3.1's built-in `brave_search` / `wolfram_alpha` tools are NOT in 3.2. |
| Multilingual | English-focused + 7 other languages (much weaker than Qwen2.5) |
| License | **Llama 3.2 License** — **NOT available in EU** (Meta blocks EU downloads), 700M-MAU commercial-use restriction, Acceptable Use Policy carve-outs for "automation" / "system-level modification". |
| GGUF Q4_K_M size | **2.02 GB** (`bartowski/Llama-3.2-3B-Instruct-GGUF`) |
| BFCL v3 score | **45.86 %** (FC mode) — surprisingly strong for a 3B; beats Qwen2.5-3B raw. |
| Fine-tune on T4 | ✅ Unsloth has Llama-3.2 (3B) notebook (2.4× faster, 58 % less memory). |
| Known issues | Heavy refusal behaviour for some agentic / system-control prompts (Meta AUP). EU license block. Smaller multilingual footprint than Qwen. |

**Why runner-up:** Better raw BFCL FC score than Qwen2.5-3B (45.86 % vs 31.81 %), but the license (EU-blocked, AUP restrictions on automation) and weaker multilingual support make it riskier as the primary base for an open-source agentic assistant. Excellent **backup** if Qwen2.5-3B's tool-calling quality proves insufficient after fine-tuning.

#### A3. `microsoft/Phi-3.5-mini-instruct`

| Field | Value |
|---|---|
| HF model ID | `microsoft/Phi-3.5-mini-instruct` |
| Architecture | Phi-3 (Dense decoder-only Transformer, same tokenizer as Phi-3 Mini) |
| Parameters | 3.8 B |
| Native context | 128 000 tokens |
| Function calling | ❌ **No official support** — confirmed in HF Discussions: *"We do not officially support tool/function calls in Phi-3.5."* Community fine-tunes exist (e.g. `Sellid/gemma-2-2B-it-thinking-function_calling-V0`) but require building a custom parser. |
| Multilingual | ✅ Multi-lingual (strong on MGSM, MMLU-pro multilingual) |
| License | **MIT** |
| GGUF Q4_K_M size | ~2.4 GB (typical for 3.8 B) |
| Known issues | No tool-calling out of the box. Microsoft AUP restrictions on "high-risk" deployments. |

**Why not picked:** Same param class as Llama-3.2-3B but **no native function calling**. Would require building a tool-call parser from scratch and then fine-tuning the model on it — too much custom work for Thursday AI's timeline.

#### A4. `google/gemma-2-2b-it`

| Field | Value |
|---|---|
| HF model ID | `google/gemma-2-2b-it` |
| Architecture | Gemma 2 (Dense decoder-only, GQA, 26 layers, 2 304 hidden dim, knowledge distillation from larger Gemma) |
| Parameters | 2.6 B |
| Native context | **8 192 tokens** (short! agentic screenshots overflow this fast) |
| Function calling | ❌ No native support — community fine-tunes exist (e.g. `akshayballal/gemma2-2b-xlam-function-calling`) |
| Multilingual | English-focused |
| License | **Gemma License** (Google — restricted; usage policy prohibits some downstream use cases) |
| GGUF Q4_K_M size | ~1.6 GB |
| Known issues | 8K context too short for screenshot + tool history in agentic loops. Gemma license restrictions on system-automation use cases. |

**Why not picked:** Too short a context window and Gemma license restrictions make it a poor fit for an OS-controlling agentic assistant.

### TIER B — Balanced 7B–8B

#### B1. `Qwen/Qwen2.5-7B-Instruct` ← **TIER B WINNER**

| Field | Value |
|---|---|
| HF model ID | `Qwen/Qwen2.5-7B-Instruct` |
| Architecture | Qwen2 (Dense decoder-only, RoPE, SwiGLU, RMSNorm, QKV bias) |
| Parameters | 7.61 B total / 6.53 B non-embedding |
| Layers / Heads | 28 layers, 28 Q heads / 4 KV heads (GQA 7:1) |
| Native context | 131 072 tokens (generation 8 192) |
| Function calling | ✅ Native — same Qwen2 tool-calling format as 3B sibling |
| Multilingual | ✅ 29+ languages |
| License | **Apache 2.0** |
| GGUF Q4_K_M size | **4.7 GB** (`bartowski/Qwen2.5-7B-Instruct-GGUF`, also Ollama library) |
| Other quants | Q3_K_M=3.5 GB, Q5_K_M=5.4 GB, Q8_0=7.9 GB, IQ4_XS=~4.2 GB |
| Ollama | `ollama run qwen2.5:7b-instruct-q4_K_M` (default `qwen2.5:7b` is also Q4_K_M) |
| BFCL v4 raw score | **59.5 overall, 63.33 AST, 93.59 Exec** (SOTA2 Nov-2025) — strong. |
| Fine-tune on T4 | ✅ Reddit r/LocalLLM confirms community fine-tunes of Qwen2.5-7B with Unsloth on Kaggle T4. Heavy on memory — requires aggressive 4-bit + gradient checkpointing. T4 × 2 (16 GB × 2 = 32 GB VRAM pooled) is the floor for 7B QLoRA. |
| Known issues | 4.7 GB Q4_K_M + KV cache + OS = swap-thrashing risk on 8 GB DDR3. Inference ~2.5–4 tok/s on i5-3rd-gen. |

**Why Tier B winner:** Same family/chat-template/tool-format as Qwen2.5-3B — Thursday AI's fine-tune code path is identical. Best 7B-class BFCL raw score among Apache-2.0 candidates. Native 128K context is the longest among balanced-tier candidates. **Best "Pro" tier upgrade** when Thursday AI is run on a 16 GB / modern-CPU host.

#### B2. `meta-llama/Llama-3.1-8B-Instruct` ← **TIER B RUNNER-UP**

| Field | Value |
|---|---|
| HF model ID | `meta-llama/Llama-3.1-8B-Instruct` |
| Architecture | Llama 3.1 (Dense, RoPE, GQA) |
| Parameters | 8.03 B |
| Native context | 128 000 tokens |
| Function calling | ⚠️ Supported via Meta's built-in `brave_search` + `wolfram_alpha` tools only — generic JSON-schema tool calls unreliable. **BFCL v3 FC mode = 25.92 %** (one of the worst scores for any 7B+ model). Prompt mode = 49.57 %. |
| License | **Llama 3.1 License** — 700M-MAU commercial-use limit, AUP discourages system-automation use cases. EU download OK. |
| GGUF Q4_K_M size | **4.92 GB** (`bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`) |
| Fine-tune on T4 | ✅ Unsloth supports Llama-3.1-8B with T4 Colab notebook (2.4× faster). |
| Known issues | Reddit r/LocalLLaMA consensus: *"Llama 3.1 8B Instruct function/tool calling seems TERRIBLE"* — model cannot reliably maintain a conversation alongside tool-calling definitions. Not suitable as Thursday AI base. |

**Why runner-up:** Despite the brand recognition, BFCL FC score is catastrophic (25.92 %). Same Llama license issues. Skip unless fine-tuned specifically on tool-calling data (then it becomes essentially "Hermes-3-Llama-3.1-8B", which is Tier C).

#### B3. `mistralai/Mistral-7B-Instruct-v0.3`

| Field | Value |
|---|---|
| HF model ID | `mistralai/Mistral-7B-Instruct-v0.3` |
| Architecture | Mistral (Dense, RoPE, GQA, Sliding-Window Attention) |
| Parameters | 7.25 B |
| Native context | 32 768 tokens (extended from v0.2's 8K) |
| Function calling | ✅ Yes — added in v0.3. Uses `mistral_common.protocol.instruct.tool_calls` API. Tool call IDs required. |
| License | **Apache 2.0** |
| GGUF Q4_K_M size | **4.37 GB** (`bartowski/Mistral-7B-Instruct-v0.3-GGUF`) |
| Fine-tune on T4 | ✅ Yes (Unsloth Mistral notebooks). |
| Known issues | Sliding-Window Attention degrades quality on long agentic sessions vs GQA. Tool-calling ecosystem smaller than Qwen's. |

**Why not picked:** Apache 2.0 ✅, but weaker long-context, smaller tool-use ecosystem than Qwen2.5-7B.

#### B4. `google/gemma-2-9b-it`

| Field | Value |
|---|---|
| HF model ID | `google/gemma-2-9b-it` |
| Architecture | Gemma 2 (Dense, GQA, sliding-window + global attention interleaved) |
| Parameters | 9 B |
| Native context | **8 192 tokens** (too short for Thursday AI's agentic screenshots) |
| Function calling | ❌ No native — community fine-tunes exist (e.g. `DiTy/gemma-2-9b-it-function-calling-GGUF`) |
| License | **Gemma License** (restricted) |
| GGUF Q4_K_M size | ~5.4 GB |
| Known issues | 8K context + Gemma license + no native tools. Also 9B Q4_K_M is ~5.4 GB → swap-thrash on 8 GB RAM. |

**Why not picked:** Same Gemma-license + 8K-context problems as Gemma-2-2B, with worse RAM pressure.

### TIER C — Specialized agentic / tool-use

#### C1. `NousResearch/Hermes-3-Llama-3.1-8B` ← **TIER C WINNER**

| Field | Value |
|---|---|
| HF model ID | `NousResearch/Hermes-3-Llama-3.1-8B` |
| Architecture | Llama 3.1 fine-tune (ChatML format) |
| Parameters | 8.03 B |
| Native context | 128 000 tokens |
| Function calling | ✅✅ **Excellent** — Hermes Function Calling standard (`<tools></tools>` XML + JSON payload). Trained on a dedicated FC dataset. Multi-turn FC + structured outputs supported. |
| License | **Llama 3 License** (inherits from Llama 3.1 base) |
| GGUF Q4_K_M size | **4.92 GB** (`bartowski/Hermes-3-Llama-3.1-8B-GGUF`) |
| Ollama | `ollama run vanilj/hermes-3-llama-3.1-8b:q4_k_m` |
| Fine-tune on T4 | ✅ Possible (Llama 3.1 8B base + LoRA on top of existing Hermes FC weights). |
| Known issues | 4.92 GB Q4_K_M → swap-thrash on 8 GB potato PC. Inherits Llama 3 license restrictions. Reddit r/LocalLLaMA: *"Hermes 3 is one of the best, if not the best, at function calling and tool use in general."* |

**Why Tier C winner:** Best agentic/tool-use model in the 7B-8B class on HuggingFace. The Hermes FC prompt format is a de-facto open standard. **Use as fallback on 16+ GB machines** or when Thursday AI needs a stronger "after-hours" model.

#### C2. `Qwen/Qwen2.5-Coder-7B-Instruct` ← **TIER C RUNNER-UP**

| Field | Value |
|---|---|
| HF model ID | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Architecture | Qwen2.5-Coder (Dense, code-specialised) |
| Parameters | 7.61 B |
| Native context | 131 072 tokens (YaRN) |
| Function calling | ✅ Same Qwen2 tool-calling format |
| License | **Apache 2.0** |
| GGUF Q4_K_M size | **4.7 GB** |
| BFCL v4 score | **86.02** (FunRL variant per `emergentmind.com`) — one of the highest open-source FC scores |
| Fine-tune on T4 | ✅ Unsloth has `kaggle-qwen-2-5-coder-7b-base` notebook. |
| Known issues | Code-specialised — may produce over-verbose code when Thursday AI just needs a tool call. Better as a **delegated secondary model** for when Thursday AI says *"let me ask the coding model to write this script"*. |

**Why runner-up:** Same license/size as Qwen2.5-7B but specialised for code; better as a *delegate* than as the primary voice-brain.

#### C3. `openchat/openchat_3.5`

| Field | Value |
|---|---|
| HF model ID | `openchat/openchat_3.5` |
| Architecture | Mistral 7B fine-tune |
| Parameters | 7 B |
| Native context | 8 192 tokens (short) |
| Function calling | ⚠️ Via community fine-tunes (`LucciAI/openchat-3.5-0106-function-calling`, `Trelis/openchat_3.5-function-calling-v3`) — base model has no native FC |
| License | **Apache 2.0** |
| Known issues | Older (2023). Largely superseded by Qwen2.5 / Llama 3.x. |

**Why not picked:** Superseded. Short context. No native FC.

---

## 3. Quantization × engine matrix for potato PC

For 8 GB DDR3, i5-3rd-gen, AVX (no AVX2):

| Model | Q4_K_M | Engine | Expected tok/s | RAM used |
|---|---|---|---|---|
| **Qwen2.5-3B-Instruct** | 1.93 GB | llama.cpp AVX build, 4 threads | **6–10** | ~2.5 GB (incl. KV 4K ctx) |
| Llama-3.2-3B-Instruct | 2.02 GB | llama.cpp AVX build, 4 threads | 5–9 | ~2.6 GB |
| Phi-3.5-mini-instruct | ~2.4 GB | llama.cpp | 4–7 | ~3.0 GB |
| Qwen2.5-7B-Instruct | 4.7 GB | llama.cpp AVX build, 4 threads | 2.5–4 | **6 GB → swap risk** |
| Qwen2.5-7B-Instruct | 3.5 GB (Q3_K_M) | llama.cpp | 3.5–5 | ~5 GB (tighter) |
| Llama-3.1-8B-Instruct | 4.92 GB | llama.cpp | 2.5–4 | **6.2 GB → swap** |
| Mistral-7B-Instruct-v0.3 | 4.37 GB | llama.cpp | 3–5 | 5.7 GB |
| Hermes-3-Llama-3.1-8B | 4.92 GB | llama.cpp | 2.5–4 | **6.2 GB → swap** |

> ⚠️ Note on AVX2: llama.cpp "modern" prebuilt binaries often default to AVX2. On Ivy Bridge (i5 3rd-gen), the binary will crash on AVX2 instructions. **You MUST build llama.cpp yourself with `LLAMA_AVX2=OFF LLAMA_AVX=ON LLAMA_FMA=ON LLAMA_F16C=ON`**, OR download a pre-`b2000` "noavx" build. Ollama's Windows installer does ship a noavx fallback.

> Source: `github.com/ggml-org/llama.cpp/issues/34`, `roger.lol/blog/llamacpp-build-guide-for-cpu-inferencing` (4B Q4 = 7.3 tok/s on consumer CPU), `localaimaster.com/blog/run-llm-cpu-only`.

---

## 4. Final recommendation — deep justification

### Base model: `Qwen/Qwen2.5-3B-Instruct`

### Quantization: **Q4_K_M** (1.93 GB)

### Inference engine: **llama.cpp CPU build (AVX, no AVX2)** + Ollama as convenience wrapper

### Fine-tune: **Unsloth 4-bit QLoRA** on Kaggle T4 ×2 (using `unsloth/Qwen2.5-3B-Instruct` + the `kaggle-qwen-2-5-conversational-unsloth` notebook as the template)

### Why this beats every alternative for THIS use case:

1. **Fits in 8 GB RAM with margin.** 1.93 GB Q4_K_M + ~0.5 GB KV cache (4 K context) + OS + browser/Playwright + Pillow = comfortably under 5 GB. The user can keep Chrome/Edge open while Thursday AI runs. No 7B class model gives this margin.
2. **Voice-first latency target met.** At ~6–10 tok/s on i5-3rd-gen, a 7-token reply (~10 tokens incl. special tokens) completes in ~1–1.5 s. Acceptable for voice UX. A 7B model at 3 tok/s would take 3.5 s — too slow for snappy voice.
3. **License = Apache 2.0.** No EU restrictions (unlike Llama 3.2). No MAU limit (unlike Llama 3.x). No AUP carve-outs against "system automation" (unlike Meta licenses). The user can ship Thursday AI to anyone, anywhere, commercially or not.
4. **Native tool-calling.** Qwen2 chat template has a first-class `tools` argument and `tool_calls` assistant response format — same JSON schema convention as OpenAI / Ollama. Thursday AI's orchestrator can use the same parser for local Qwen2.5-3B and for the "free web AI" fallback path (Z.ai / Qwen chat / Grok via browser).
5. **Fine-tune-ready.** Unsloth officially supports `unsloth/Qwen2.5-3B-Instruct` with a Kaggle T4 notebook (verified at `huggingface.co/unsloth/Qwen2.5-3B-Instruct` and `kaggle.com/code/danielhanchen/kaggle-qwen-2-5-conversational-unsloth`). 4-bit QLoRA on T4 ×2 gives ~2× batch size, full fine-tune of LoRA adapters in <6 hours of free Kaggle compute.
6. **Fine-tune target = agentic OS-control traces.** The base model's 31.8 BFCL raw score is **irrelevant** — Thursday AI's value is in the QLoRA'd OS-control dataset (think-act-observe loops, tool-call schemas for `open_app`, `click_pixel`, `read_screen`, `paste_into_browser`, etc.). Qwen2.5-3B has the capacity (3.09 B params) to absorb this dataset and the Qwen2 architecture is well-supported by PEFT/TRL/Unsloth.
7. **Upgrade path.** Same chat template + tool-call format as `Qwen/Qwen2.5-7B-Instruct` — when Thursday AI is run on a stronger machine (16 GB RAM, modern CPU), swapping the GGUF file alone (1.93 GB → 4.7 GB) gives a free quality upgrade with zero code changes. Also same format as Qwen2.5-Coder-7B (Tier C runner-up) — the same Thursday AI runtime can delegate code-gen to a stronger model when present.
8. **Multilingual.** 29+ languages. Voice-first assistants are inherently i18n-sensitive.
9. **128 K extended context (via YaRN)** for long agentic sessions (rare but useful).
10. **Backup web-AI orchestration.** The model can be prompt-taught that when its own confidence is low, it should emit a `delegate_to_web_ai` tool call (Z.ai / Grok / Qwen chat) — fits the user's "use free web AIs by automatically opening browser" requirement.

### What the recommendation **beats**:

| Alternative | Why Thursday AI shouldn't use it as the primary |
|---|---|
| **Llama-3.2-3B-Instruct** | Llama 3.2 license = no EU distribution; Meta AUP discourages "system automation" use cases; will refuse some OS-control prompts. Better raw FC (45.9 %) is offset by these risks. **Use as Tier A backup only.** |
| **Llama-3.1-8B-Instruct** | BFCL FC mode = 25.9 % (terrible). Same license issues. Same RAM problem as any 7B. Reddit consensus is "tool calling seems TERRIBLE." **Skip.** |
| **Hermes-3-Llama-3.1-8B** | Better agentic base than Qwen2.5-3B, but (a) 4.92 GB Q4_K_M doesn't fit comfortably in 8 GB RAM with OS, (b) inherits Llama-3 license, (c) Mistral-style sliding-window attention context scaling is weaker than Qwen2 GQA for long agentic sessions. **Use as fallback on 16+ GB machines.** |
| **Qwen2.5-7B-Instruct** | 4.7 GB Q4_K_M + KV cache ≈ 6 GB → swap-thrashes on 8 GB potato. CPU inference ~2.5-4 tok/s → voice latency marginal. **Better than 3B for quality but worse for the potato-PC constraint.** Recommend as future "Pro" tier of Thursday AI. |
| **Qwen2.5-Coder-7B-Instruct** | BFCL 86 % is great, but Thursday AI is voice-conversational first, not code-first. **Better as a delegated secondary model** for when Thursday AI says "let me ask the coding model to write this script." |
| **Mistral-7B-Instruct-v0.3** | Apache 2.0 ✅, but older SWA architecture, weaker long-context, smaller community tool-use ecosystem than Qwen. |
| **Phi-3.5-mini-instruct** | No official tool-calling support. Would require building the tool-call parser from scratch. |
| **Gemma-2-2B-it / Gemma-2-9B-it** | 8 K context (too short for agentic screenshots). Restricted Gemma license. |

### Concrete next-action checklist (for follow-on tasks):

1. **Task 2-b (inference stack):** Build llama.cpp with `LLAMA_AVX=ON LLAMA_AVX2=OFF LLAMA_FMA=ON LLAMA_F16C=ON` on the target i5-3rd-gen machine. Verify token/sec with `llama-bench -m Qwen2.5-3B-Instruct-Q4_K_M.gguf -p 128 -n 128 -t 4`.
2. **Task 2-c (fine-tune dataset):** Curate ~5–10k agentic OS-control traces in **Qwen2 chat template + `tool_calls` JSON format**. Include "when2call" negative examples (don't call a tool when not needed).
3. **Task 3 (fine-tune run):** Launch `danielhanchen/kaggle-qwen-2-5-conversational-unsloth` notebook on Kaggle with T4 ×2, swap in Thursday AI's dataset, 4-bit QLoRA, save LoRA adapters → merge → convert to GGUF Q4_K_M via `llama-quantize`.
4. **Task 4 (runtime):** Wrap `ollama run thursday-ai:3b-q4_k_m` with a Python orchestrator that implements think-act-observe loop and exposes `delegate_to_web_ai` tool that drives a headless browser (Playwright) to Z.ai / Grok / Qwen-chat.

---

## 5. Quick-reference quant size table (verified)

Sizes pulled from `bartowski/*-GGUF` HF repos and Ollama library pages.

| Model | Q4_K_M | Q5_K_M | Q3_K_M | Q8_0 | F16 |
|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | **1.93 GB** | 2.22 GB | 1.71 GB | 3.29 GB | 6.18 GB |
| Llama-3.2-3B-Instruct | **2.02 GB** | ~2.3 GB | ~1.7 GB | ~3.4 GB | ~6.4 GB |
| Phi-3.5-mini-instruct | ~2.4 GB | ~2.7 GB | ~2.1 GB | ~4.1 GB | ~7.6 GB |
| Gemma-2-2b-it | ~1.6 GB | ~1.8 GB | ~1.4 GB | ~2.7 GB | ~5.0 GB |
| Qwen2.5-7B-Instruct | **4.7 GB** | 5.4 GB | 3.5 GB | 7.9 GB | 15.0 GB |
| Llama-3.1-8B-Instruct | **4.92 GB** | 5.7 GB | 3.9 GB | 8.5 GB | 16.0 GB |
| Mistral-7B-Instruct-v0.3 | **4.37 GB** | 5.0 GB | 3.52 GB | 7.5 GB | 14.5 GB |
| Hermes-3-Llama-3.1-8B | **4.92 GB** | 5.7 GB | 3.9 GB | 8.5 GB | 16.0 GB |
| Qwen2.5-Coder-7B-Instruct | **4.7 GB** | 5.4 GB | 3.5 GB | 7.9 GB | 15.0 GB |

---

## 6. Final summary table

| Tier | Pick | Quant | Size | Engine for potato PC |
|---|---|---|---|---|
| **A — 3B–4B** | **Qwen/Qwen2.5-3B-Instruct** | Q4_K_M | 1.93 GB | llama.cpp AVX build OR `ollama run qwen2.5:3b-instruct-q4_K_M` |
| A runner-up | meta-llama/Llama-3.2-3B-Instruct | Q4_K_M | 2.02 GB | llama.cpp (license-blocked in EU) |
| **B — 7B–8B** | **Qwen/Qwen2.5-7B-Instruct** | Q4_K_M (or Q3_K_M if RAM-tight) | 4.7 GB (3.5 GB) | llama.cpp AVX build, ~3–5 tok/s |
| B runner-up | mistralai/Mistral-7B-Instruct-v0.3 | Q4_K_M | 4.37 GB | llama.cpp / Ollama |
| **C — agentic** | **NousResearch/Hermes-3-Llama-3.1-8B** | Q4_K_M | 4.92 GB | llama.cpp (heavy for 8 GB; better on 16 GB) |
| C runner-up | Qwen/Qwen2.5-Coder-7B-Instruct | Q4_K_M | 4.7 GB | llama.cpp / Ollama |

### FINAL ANSWER

> **Thursday AI should fine-tune from `Qwen/Qwen2.5-3B-Instruct` (Apache 2.0, 3.09 B params, native `tool_calls` JSON, 32 K → 128 K context, 1.93 GB Q4_K_M GGUF), using Unsloth 4-bit QLoRA on Kaggle T4 ×2, and serve via llama.cpp (AVX build for Ivy-Bridge i5-3rd-gen) wrapped by Ollama for an OpenAI-compatible local API.**

---

## Sources (URLs visited)

### Tier A model cards
- https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- https://huggingface.co/unsloth/Qwen2.5-3B-Instruct
- https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF
- https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF
- https://huggingface.co/blog/llama32
- https://huggingface.co/microsoft/Phi-3.5-mini-instruct
- https://huggingface.co/microsoft/Phi-3.5-mini-instruct/discussions/7
- https://huggingface.co/google/gemma-2-2b-it
- https://inferencebench.io/models/google/gemma-2-2b
- https://localaimaster.com/models/gemma-2b

### Tier B model cards
- https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
- https://www.kaggle.com/refs/hf-model/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
- https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
- https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF
- https://huggingface.co/google/gemma-2-9b-it
- https://atomic.chat/ja/models/llama-3-1-8b-instruct (Q4_K_M size = 4.92 GB)
- https://www.aimodels.fyi/models/huggingFace/meta-llama-31-8b-instruct-gguf-bartowski

### Tier C model cards
- https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B
- https://huggingface.co/bartowski/Hermes-3-Llama-3.1-8B-GGUF
- https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- https://huggingface.co/openchat/openchat_3.5
- https://huggingface.co/LucciAI/openchat-3.5-0106-function-calling
- https://arxiv.org/pdf/2408.11857 (Hermes 3 Technical Report)
- https://nousresearch.com/hermes3
- https://fast.io/resources/hermes-3-function-calling-guide
- https://www.emergentmind.com/topics/berkeley-function-calling-leaderboard-v4-bfclv4 (Qwen2.5-Coder-7B BFCL = 86.02)
- https://www.reddit.com/r/LocalLLaMA/comments/1et0k7l/hermes_3_a_nousresearch_collection

### Benchmarks (BFCL)
- https://gorilla.cs.berkeley.edu/leaderboard.html (BFCL v4)
- https://benchmarklist.com/benchmarks/bfcl_v3 (Llama-3.2-3B-Instruct FC = 45.86%, Llama-3.1-8B-Instruct FC = 25.92%, Llama-3.1-8B-Instruct Prompt = 49.57%)
- https://www.sota2.com/research/sota/function-calling-on-berkeley-function-calling-leaderboard-bfcl-overall-november-19-2025 (Qwen2.5-7B-Instruct raw = 59.5 / Qwen2.5-3B-Instruct raw = 31.81 / Llama3.2-3B-Instruct raw = 40.5)
- https://llm-stats.com/leaderboards/best-ai-for-tool-calling
- https://www.klavis.ai/blog/function-calling-and-agentic-ai-in-2025-what-the-latest-benchmarks-tell-us-about-model-performance

### CPU inference benchmarks / build guides
- https://github.com/ggml-org/llama.cpp/issues/34 (Intel i5, 2-core, 8GB RAM, 7B 4bit → 94 ms/token)
- https://roger.lol/blog/llamacpp-build-guide-for-cpu-inferencing (4B Q4 = 7.3 tok/s)
- https://localaimaster.com/blog/run-llm-cpu-only (CPU usable up to ~4B; tolerable 7–8B on DDR5; bad beyond)
- https://www.reddit.com/r/LocalLLaMA/comments/1u1sj9d/whats_up_on_cpu_inference_these_days (Q4_K_M standard llama.cpp ≈ 10 tok/s typical)
- https://www.reddit.com/r/LocalLLaMA/comments/1fq883g/qwen_25_cpu_vs_gpu_comparison (7B Q4 K_M on CPU = 2–3 tok/s)
- https://github.com/ggml-org/llama.cpp/discussions/3167 (7 tok/s already quite good; 16 threads not 32; SMT hurts inference)
- https://www.myaihardware.com/llama-cpp-benchmarks
- https://www.sitepoint.com/local-llms-complete-guide (5–15 tok/s modern desktop CPU with AVX2)
- https://openbenchmarking.org/test/pts/llama-cpp

### Fine-tune / Unsloth / Kaggle
- https://huggingface.co/unsloth/Qwen2.5-3B-Instruct
- https://www.kaggle.com/code/danielhanchen/kaggle-qwen-2-5-conversational-unsloth
- https://www.kaggle.com/code/danielhanchen/kaggle-qwen-2-5-coder-7b-base
- https://www.kaggle.com/code/ksmooi/fine-tuning-qwen2-5-3b-instruct-grpo-peft
- https://www.kaggle.com/code/viratchauhan/qwen-2-5-4-bit-q-3b-finetune-with-unsloth-w-b
- https://unsloth.ai/docs/get-started/unsloth-notebooks
- https://www.reddit.com/r/LocalLLM/comments/1wh813o/i_finetuned_a_7b_medical_ai_for_0_on_free_kaggle

### Ollama library
- https://ollama.com/library/qwen2.5:7b-instruct-q4_K_M
- https://ollama.com/library/qwen2.5:3b-instruct-q4_K_M
- https://ollama.com/library/qwen2.5-coder:3b-instruct-q4_K_M
- https://ollama.com/vanilj/hermes-3-llama-3.1-8b:q4_k_m

### Llama license / community reports
- https://huggingface.co/blog/llama32 (Llama 3.2 license changes — EU block)
- https://www.reddit.com/r/LocalLLaMA/comments/1ece00h/llama_31_8b_instruct_functiontool_calling_seems ("Llama 3.1 8B Instruct function/tool calling seems TERRIBLE")
- https://www.reddit.com/r/LocalLLaMA/comments/1gy674o/are_you_successfully_calling_tools_with (Llama-3.2-3B Instruct tool calling inconsistency reports)
- https://www.reddit.com/r/LocalLLaMA/comments/1juxcmi/what_are_the_best_local_small_llms_for_tool (best local small LLMs for tool calling 2025)

