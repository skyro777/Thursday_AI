# Thursday AI

> An **offline, voice-first, autonomous** AI assistant that controls your PC. You talk; it acts — opening apps, driving the browser, taking screenshots, delegating to heavier free web AIs (Z.ai / Grok / Qwen-chat) when needed. Runs locally on a potato PC (8 GB DDR3, i5-3rd-gen, Intel HD graphics). Privacy is total: nothing leaves your machine unless Thursday itself opens a browser on your behalf.

**Status:** v0.1 — research complete, plan locked, dataset generator + Kaggle training notebook ready, awaiting first smoke-train (M1).

---

## Quick links

- 🧭 **[LIVING_PLAN.md](LIVING_PLAN.md)** — the evolving master plan (start here)
- 🏗️ **[ARCHITECTURE.md](ARCHITECTURE.md)** — system design + data flow
- 🔬 **[research/](research/)** — three deep research reports:
  - [01 — Base model selection](research/01_base_model_selection.md) → `Qwen2.5-3B-Instruct` @ `Q4_K_M`
  - [02 — Training data approach](research/02_training_data_approach.md) → Qwen2 tool-call JSONL, 26 tools, 90k examples
  - [03 — Kaggle T4×2 QLoRA recipe](research/03_finetuning_recipe_kaggle.md) → Unsloth, batch 16, 3 epochs
- 🧠 **[training/thursday_ai_finetune.ipynb](training/thursday_ai_finetune.ipynb)** — the Kaggle notebook
- 🛠️ **[data/schema/tools.json](data/schema/tools.json)** — the 26-tool action vocabulary

---

## What Thursday AI is

| | |
|---|---|
| **Voice-first** | Wake word "Hey Thursday" → ASR → LLM → TTS. Hands-free by default. |
| **Short-while-working** | While a task is in progress, replies are 3-7 words. Final summary ≤ 2 lines unless asked for more. |
| **Offline by default** | The LLM is `Qwen2.5-3B-Instruct` fine-tuned via QLoRA, served via `llama.cpp` or `Ollama`. No cloud inference. |
| **Agentic OS control** | 26 tools cover window management, mouse/keyboard, screenshots, filesystem, browser, web-AI delegation, and OS conveniences (alarm, volume, notifications). |
| **Privacy-respecting** | Zero telemetry. Personal files never leave the machine. Destructive actions need voice confirmation. |
| **Delegates when smart** | For tasks beyond a 3B model (writing complex code, analyzing images), it drives a browser to a free web AI (Z.ai, Grok, Qwen-chat) or calls `paste_to_webai` headlessly. |
| **Potato-friendly** | Target hardware: i5-3rd-gen + 8 GB DDR3 + Intel HD graphics. Q4_K_M GGUF = 1.93 GB; expected ~6-10 tok/s. |

---

## The model decision (locked v0.1)

| Aspect | Decision |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` (3.09 B params, Apache-2.0, 32K native ctx → 128K via YaRN, native `tool_calls` JSON) |
| Fine-tune | Unsloth 4-bit QLoRA on Kaggle T4 ×2 |
| Deployment quant | GGUF `Q4_K_M` (1.93 GB) |
| Inference engine | `llama.cpp` (CPU, AVX1 build, no AVX2) **or** `Ollama` (auto noavx fallback) |
| Tool-call format | Qwen2 chat template (Hermes variant) |
| Upgrade path | Drop-in swap to `Qwen2.5-7B-Instruct` on a 16 GB+ host |

Full reasoning: `research/01_base_model_selection.md`.

---

## The dataset (v0.1 spec)

Each training example is a JSONL line with `{system, tools, messages}` in the Qwen2 tool-call format. Each `assistant` turn has at most ONE `tool_calls` entry (atomicity rule). Each `tool_call.id` is echoed back by a `tool`-role observation.

**Sample trajectory** (`data/examples/alarm_7am.jsonl`):

```json
{"system": "You are Thursday... CURRENT OS: Linux (GNOME)...",
 "tools": [{"name":"set_alarm",...}, {"name":"ask_user",...}, {"name":"finish",...}],
 "messages": [
   {"role":"user","content":"Hey Thursday, set an alarm for 7 am tomorrow."},
   {"role":"assistant","content":"On it.","tool_calls":[{"id":"call_001","type":"function","function":{"name":"set_alarm","arguments":"{\"hour\":7,\"minute\":0,\"label\":\"tomorrow 7am\"}"}}]},
   {"role":"tool","name":"set_alarm","content":"{\"ok\":true,\"alarm_id\":\"a4f9\",...}"},
   {"role":"assistant","content":"Done.","tool_calls":[{"id":"call_002","type":"function","function":{"name":"finish","arguments":"{\"summary\":\"Alarm set for 07:00 tomorrow.\"}"}}]}
 ]}
```

**Composition target (90,000 examples for v1):**

| Source | Examples | % |
|---|---|---|
| Template-based (Method A) | 25,000 | 28% |
| LLM-distilled Evol-Instruct (Method B) | 35,000 | 39% |
| LLM-distilled amplification of recorded traces | 12,000 | 13% |
| Real Playwright/pyautogui recordings (Method C) | 5,000 | 6% |
| xLAM-function-calling-60k filtered | 8,000 | 9% |
| Mind2Web converted | 4,000 | 4% |
| WebVoyager converted | 1,000 | 1% |

Full spec: `research/02_training_data_approach.md`.

---

## How to use this repo

### 1. Generate the starter dataset (Phase 0 — smoke test, 1k examples)

```bash
# From the repo root
python scripts/generate_template_data.py --count 1000 --out data/processed/template_generated.jsonl --seed 42
python scripts/validate_dataset.py data/processed/template_generated.jsonl data/examples/*.jsonl
```

### 2. (Optional) Generate LLM-distilled examples (Method B)

Requires either a local Ollama server or a `ZAI_API_KEY`:

```bash
# Using local Ollama (free, offline)
ollama serve &
ollama pull qwen2.5:7b-instruct-q4_K_M
python scripts/generate_llm_distilled.py --count 5000 --seeds data/examples --teacher ollama

# Using ZAI SDK (free tier, requires ZAI_API_KEY env var)
export ZAI_API_KEY=...
python scripts/generate_llm_distilled.py --count 5000 --seeds data/examples --teacher zai
```

### 3. Train on Kaggle T4 ×2

1. Upload `training/thursday_ai_finetune.ipynb` to Kaggle as a new notebook.
2. Set accelerator: **GPU T4 x2**.
3. Add Kaggle Secret: `HF_TOKEN` (HuggingFace write token).
4. Run all cells. ~6-8 hours for 90k examples × 3 epochs.

The notebook produces:
- `thursday-ai-v0.1-Q4_K_M.gguf` (~1.93 GB)
- Push to HuggingFace Hub at `skyro777/thursday-ai-v0.1-gguf` (configurable in the notebook)

### 4. Run on your potato PC

```bash
# Option A: Ollama (recommended)
ollama pull hf.co/skyro777/thursday-ai-v0.1-gguf:Q4_K_M
ollama run hf.co/skyro777/thursday-ai-v0.1-gguf:Q4_K_M

# Option B: llama.cpp direct (slightly faster)
# Build llama.cpp with AVX1 (no AVX2) for i5 3rd-gen:
cmake -B build -DLLAMA_AVX2=OFF -DLLAMA_AVX=ON -DLLAMA_FMA=ON -DLLAMA_F16C=ON
cmake --build build --config Release
./build/bin/llama-server --model thursday-ai-v0.1-Q4_K_M.gguf --threads 4 --ctx-size 4096 --port 8080
```

The runtime orchestrator (M3 work) will glue Whisper.cpp + this LLM + Piper TTS + the 26 tool implementations into a hands-free voice loop. Watch the `LIVING_PLAN.md` roadmap for status.

---

## Repo layout

```
Thursday_AI/
├── README.md                      ← this file
├── LIVING_PLAN.md                 ← the evolving master plan
├── ARCHITECTURE.md                ← system design + data flow
├── research/                      ← deep research reports
│   ├── 01_base_model_selection.md
│   ├── 02_training_data_approach.md
│   └── 03_finetuning_recipe_kaggle.md
├── data/
│   ├── schema/
│   │   └── tools.json             ← the 26 tool definitions
│   ├── system_prompts/
│   │   └── thursday_default.md    ← system prompt template
│   ├── examples/                  ← hand-written golden JSONL examples
│   │   ├── alarm_7am.jsonl
│   │   ├── find_youtube_video.jsonl
│   │   ├── summarize_screen.jsonl
│   │   ├── world_news.jsonl
│   │   ├── explain_script.jsonl
│   │   └── volume_and_open.jsonl
│   └── processed/                 ← generated datasets (gitignored except for samples)
├── scripts/
│   ├── generate_template_data.py  ← Method A generator
│   ├── generate_llm_distilled.py  ← Method B generator (Evol-Instruct)
│   ├── validate_dataset.py        ← JSON-schema + tool-call-id validator
│   └── merge_and_quantize.py      ← post-train merge + GGUF convert
├── training/
│   ├── thursday_ai_finetune.ipynb ← Kaggle T4×2 notebook
│   ├── requirements.txt
│   └── README.md
├── runtime/                       ← placeholder (M3 work)
└── eval/                          ← placeholder (M4 work)
```

---

## Roadmap (high level — see `LIVING_PLAN.md` for detail)

- **M0 — Foundation** *(current)*: research, plan, dataset spec, training notebook ✅
- **M1 — Smoke dataset (1k)**: train v0.1-smoke, sanity-test on a real PC
- **M2 — Full dataset (90k)**: train v0.1, hit ≥ 50% on `thursday_eval_v1`
- **M3 — Voice + runtime**: Python orchestrator wiring Whisper → LLM → tools → Piper
- **M4 — Eval harness**: 50-task cross-OS benchmark
- **M5 — GUI app (v2)**: Tauri installer with tray icon
- **M6 — Memory + personalisation (v2.5)**: local profile, long-term memory
- **M7 — Multimodal vision (v3)**: plug a free VLM for screen understanding

---

## License

MIT (see `LICENSE`). The fine-tuned model is derived from `Qwen2.5-3B-Instruct` (Apache-2.0) and inherits that license. The 26-tool action vocabulary and system prompts are original to this project and MIT-licensed.

## Contributing

This is a personal project by [@skyro777](https://github.com/skyro777). For now, file issues for discussion; PRs welcome after M1.

## Acknowledgements

Built with:
- [Qwen2.5](https://qwenlm.github.io/) — base model
- [Unsloth](https://github.com/unslothai/unsloth) — QLoRA fine-tuning acceleration
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — CPU inference
- [Ollama](https://ollama.com) — easy model distribution
- [z-ai-web-dev-sdk](https://github.com/zai-org/z-ai-web-dev-sdk) — headless web-AI delegation
- Inspiration: [Open Interpreter](https://github.com/OpenInterpreter/open-interpreter), [Claude Computer Use](https://www.anthropic.com/news/3-5-models-and-computer-use), [UFO](https://github.com/microsoft/UFO), [Self-Operating Computer](https://github.com/OthersideAI/self-operating-computer), [OSWorld](https://os-world.github.io/)
