# Thursday AI — Living Plan

> **This is a LIVING document.** Every time we change direction, hit a milestone, learn something new, or revise scope, we update this file and commit. The newest state is the source of truth. Older decisions are kept under `## Decision History` so we remember *why* we are where we are.
>
> Owner: @skyro777 (with GLM-5.2 as the build copilot)
> Repo: https://github.com/skyro777/Thursday_AI
> Started: 2025
> Current revision: **v0.1 (kickoff)**

---

## 0. The One-Paragraph Pitch

**Thursday AI** is an *offline, voice-first, autonomous* personal AI that lives on your PC and controls it for you. You talk to it ("Hey Thursday, find that canoe-building YouTube video I described last week") and it acts — opening apps, driving the browser, taking screenshots, delegating to heavier free web AIs (Z.ai, Grok, Qwen Chat) when it needs more brain, and replying in 3–7 words while working with a one-line summary when done. It runs locally on a potato PC (8 GB DDR3, i5-3rd-gen, Intel HD graphics), is fine-tuned from `Qwen/Qwen2.5-3B-Instruct` with QLoRA on Kaggle T4×2, and learns its skill — *how to operate your computer* — from a curated + synthetic dataset of agentic trajectories in the Qwen2 tool-call format. Privacy is total: nothing leaves the machine except traffic you'd generate yourself browsing the web.

---

## 1. North-Star Goals (what "done" looks like for v1.0)

| # | Goal | Measurable success criterion |
|---|------|------------------------------|
| G1 | **Offline by default** | All voice recognition + LLM inference runs on the user's CPU with no internet. Web access is *only* when Thursday AI itself decides to drive a browser on the user's behalf. |
| G2 | **Runs on a potato PC** | Cold-start → first audio reply in < 8 s on i5-3rd-gen + 8 GB DDR3 + Intel HD. Steady-state reply latency < 1.5 s for a 7-word response. |
| G3 | **Voice-first** | Wake word "Hey Thursday" → ASR → LLM → TTS pipeline with no click required. Hands-free operation is the default. |
| G4 | **Agentic OS control** | Can autonomously complete 80 % of the 50-task `thursday_eval_v1` benchmark (alarm-setting, app launching, browser search, file ops, screenshot understanding, web-AI delegation). |
| G5 | **Short-while-working** | While a multi-step task is in progress, ≥ 70 % of the model's turns are ≤ 7 words. Final summary is ≤ 2 lines unless asked for more. |
| G6 | **Free-tier-power-savvy** | Knows how to use free web AIs (Z.ai chat, Grok free, Qwen-chat free, Bing Copilot free) by *driving the browser like a human*, including paste-prompt → read-response → continue. |
| G7 | **Privacy-respecting** | Zero telemetry. No model output is logged anywhere outside the user's machine unless they explicitly enable a debug log. |
| G8 | **Standalone GUI app (later)** | Packaged as an installer (Windows `.exe`, Linux `.AppImage`, macOS `.dmg`) with a tray icon — but only *after* the model itself is solid (deferred to v2). |

---

## 2. Scope of v1 (this plan)

**In scope:**
- Choose base model + quant + inference engine ✅ (decided — see §4)
- Design the action vocabulary / tool surface ✅ (decided — see §5)
- Design the dataset schema + produce a starter dataset (≈ 1k examples) + a generator that scales to 90k ✅ (in progress)
- Produce the Kaggle T4×2 fine-tuning notebook that turns Qwen2.5-3B-Instruct into Thursday AI v0.1 ✅ (in progress)
- Define a 50-task evaluation benchmark (`thursday_eval_v1`)
- Document the future runtime architecture (so we know what the model must support)

**Explicitly out of scope for v1:**
- The standalone GUI app (electron / tauri / pywebview) — v2
- ASR / TTS integration with the model — v1 ships the model + a Python orchestrator that calls Whisper.cpp + Piper + llama.cpp; the GUI is layered on later
- Continuous online learning / memory graph — v2
- Mobile companion — v3
- Multi-user / cloud sync — never (privacy by design)

---

## 3. The Big Picture (system map)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          USER (talks / listens)                            │
└──────────────┬─────────────────────────────────────────────┬──────────────┘
               │ voice in                                    │ voice out
               ▼                                             ▲
      ┌────────────────┐                            ┌────────────────┐
      │  Whisper.cpp   │  → text                    │   Piper TTS    │
      │  (small.en)    │                             │  (VCTK voice)  │
      └───────┬────────┘                            └────────┬───────┘
              │ text                                          │ text
              ▼                                               │
      ┌────────────────────────────────────────────────────────┴──────┐
      │              Thursday Orchestrator (Python)                    │
      │  • thinks → emits ONE tool_call → waits for observation →      │
      │    loops until goal complete or asks user                      │
      │  • enforces the 3-7-word "working" reply style                 │
      │  • human-in-the-loop confirmation for destructive actions      │
      └──────┬─────────────────────────────────────────┬──────────────┘
             │ HTTP (localhost)                       │ subprocess
             ▼                                        ▼
      ┌──────────────────┐                  ┌──────────────────────┐
      │  llama.cpp /     │                  │  Tool Runtime (Python)│
      │  Ollama server   │                  │  • pyautogui (mouse/  │
      │  (Qwen2.5-3B     │◄── stream ──────│    kb)                │
      │   Thursday-LoRA  │                  │  • playwright (browser)│
      │   Q4_K_M)        │                  │  • subprocess (shell) │
      └──────────────────┘                  │  • mss (screenshots)  │
                                            │  • tesseract (OCR)    │
                                            │  • z-ai-web-dev-sdk   │
                                            │    (web_search +      │
                                            │     web_reader)       │
                                            └──────────────────────┘
```

Key idea: the LLM is the brain. The Python orchestrator is the hands. The LLM emits structured `tool_calls`; the orchestrator executes them and feeds observations back. Voice + screenshots are just another pair of tools.

---

## 4. The Model Decision (locked in v0.1)

| Aspect | Decision | Why |
|---|---|---|
| **Base model** | `Qwen/Qwen2.5-3B-Instruct` | Fits 8 GB RAM at Q4_K_M (1.93 GB) with headroom; Apache-2.0; native `tool_calls` JSON; same template as the 7B sibling → free upgrade path on stronger machines; Unsloth has a Kaggle T4 fine-tune notebook ready; 29+ languages. Full reasoning in `research/01_base_model_selection.md`. |
| **Fine-tune method** | Unsloth 4-bit QLoRA on Kaggle T4×2 | 32 GB pooled VRAM handles a 3B model with full LoRA r=32, seq_len 4096, batch 4×grad-accum 8. Single 9-hr session is enough for 90k examples × 3 epochs. Recipe in `research/03_finetuning_recipe_kaggle.md`. |
| **Quant for deployment** | GGUF `Q4_K_M` (1.93 GB) | Best quality/size ratio for Qwen2.5-3B; runs on AVX-only CPUs (i5-3rd-gen has AVX but not AVX2). |
| **Inference engine** | `llama.cpp` (CPU, AVX1 build, no AVX2) OR `Ollama` (which auto-falls-back to noavx on Windows) | llama.cpp is ~30 % faster than Ollama in headless benchmarks but Ollama is easier to install. Ship both; default to Ollama for the GUI later. |
| **Tool-call format** | Qwen2 chat template (Hermes variant) with `<tools>...</tools>` block + `tool_calls` JSON in assistant messages | Native to the base model → minimal template drift after fine-tune. |
| **Upgrade path** | Drop-in swap to `Qwen/Qwen2.5-7B-Instruct` on a 16 GB+ machine | Same chat template, same tool format, just a bigger GGUF. |

---

## 5. The Action Vocabulary (locked in v0.1)

Thursday AI exposes **26 platform-agnostic tools** to the model. The runtime translates each call into OS-specific primitives (`xdotool` on Linux, `pyautogui` cross-platform, AppleScript on macOS, UI Automation API on Windows). The full JSON-schema definitions live in `data/schema/tools.json`. Summary:

- **Window manager** (4): `open_app`, `close_window`, `switch_app`, `minimize_window`
- **Mouse/keyboard** (5): `click`, `drag`, `type_text`, `key_press`, `scroll`
- **Screenshot/vision** (1): `screenshot` (returns path + OCR text + window title)
- **Filesystem** (4): `list_dir`, `read_file`, `write_file`, `run_shell`
- **Browser** (4): `browser_open`, `browser_search`, `browser_navigate`, `browser_close_tab`
- **Web-AI delegation** (2): `paste_to_webai`, `read_webai_response` (uses `z-ai-web-dev-sdk`)
- **OS convenience** (5): `set_alarm`, `set_volume`, `toggle_mute`, `open_url`, `notify_user`
- **Conversation/safety** (2): `ask_user`, `finish`

Design rules baked into the vocabulary:
1. **Atomicity** — one tool call per assistant turn. The model thinks, calls one tool, waits, repeats.
2. **Destructive actions need user confirmation** — the orchestrator intercepts `write_file`, `run_shell`, `close_window` and asks the user (voice) before executing.
3. **Screenshots are first-class** — `screenshot` returns OCR text so the model can reason without vision (and we can later attach the PNG to a free VLM via `paste_to_webai`).
4. **Web-AI delegation is explicit** — the model must *choose* to delegate (`paste_to_webai`) rather than hallucinate facts.

---

## 6. The Dataset (v0.1 spec)

**Schema:** Qwen2 tool-call chat template. Each example is one JSONL line:
```json
{"system": "...", "tools": [...], "messages": [
  {"role":"user","content":"..."},
  {"role":"assistant","content":"...","tool_calls":[{"id":"call_001","type":"function","function":{"name":"...","arguments":"{...}"}}]},
  {"role":"tool","name":"...","content":"{...}"},
  {"role":"assistant","content":"final summary"}
]}
```

**Composition (target 90,000 examples):**

| Source | Examples | % |
|---|---|---|
| Template-based (Method A) — `scripts/generate_template_data.py` | 25,000 | 28 % |
| LLM-distilled Evol-Instruct (Method B) — `scripts/generate_llm_distilled.py` | 35,000 | 39 % |
| LLM-distilled amplification of recorded traces | 12,000 | 13 % |
| Real Playwright/pyautogui recordings (Method C) | 5,000 | 6 % |
| xLAM-function-calling-60k filtered subset | 8,000 | 9 % |
| Mind2Web converted to our schema | 4,000 | 4 % |
| WebVoyager converted | 1,000 | 1 % |

**Phase plan:**

| Phase | Examples | Epochs | Purpose |
|---|---|---|---|
| Phase 0 — smoke test | 1,000 | 1 | Verify pipeline end-to-end |
| Phase 1 — format acquisition | 15,000 | 2 | Teach tool-call format + 26-tool vocabulary |
| Phase 2 — generalisation | 90,000 | 3 | The full mixture |

Full spec in `research/02_training_data_approach.md`.

---

## 7. Build Roadmap

> Phases are sequential within a milestone but milestones can overlap. Each milestone ends with a commit + tag + (when relevant) a HuggingFace upload.

### Milestone M0 — Foundation (current)
**Goal:** repo, plan, research, dataset spec, model decision locked.

- [x] Clone repo, set up workspace
- [x] Research: base model (Task 2-a) → `research/01_base_model_selection.md`
- [x] Research: training-data approach (Task 2-b) → `research/02_training_data_approach.md`
- [x] Research: Kaggle QLoRA recipe (Task 2-c) → `research/03_finetuning_recipe_kaggle.md`
- [x] Lock model decision (Qwen2.5-3B-Instruct, Q4_K_M, llama.cpp/Ollama)
- [x] Lock tool vocabulary (26 tools)
- [x] Lock dataset schema (Qwen2 tool-call JSONL)
- [ ] Write LIVING_PLAN.md (this file) ← *in progress*
- [ ] Write ARCHITECTURE.md
- [ ] Write the 26-tool JSON definitions (`data/schema/tools.json`)
- [ ] Write starter template-based generator (`scripts/generate_template_data.py`)
- [ ] Write 5 sample JSONL training examples (`data/examples/`)
- [ ] Write the Kaggle notebook (`training/thursday_ai_finetune.ipynb`)
- [ ] Write dataset validator (`scripts/validate_dataset.py`)
- [ ] Write README
- [ ] Commit + push M0

### Milestone M1 — Smoke dataset (Phase 0)
**Goal:** 1,000 high-quality examples, train a smoke-checkpoint, sanity-test on a potato-PC.

- [ ] Run template generator → ~500 examples
- [ ] Hand-author 20 "golden" examples covering every tool at least once
- [ ] Use a free LLM (Z.ai chat via SDK, or Qwen-chat) to distil ~480 more
- [ ] Validate dataset (every `tool_call.id` matched by a `tool` response, every tool name in `tools`)
- [ ] Upload dataset to HF Hub as `thursday-ai/thursday-os-control-v0.1-smoke`
- [ ] Run Kaggle notebook → produce `thursday-ai-v0.1-smoke` GGUF
- [ ] Download GGUF, run on a real or simulated potato PC, manually try 10 tasks
- [ ] Document failures, fix dataset, iterate

### Milestone M2 — Full dataset (Phase 2)
**Goal:** 90,000-example mixture, train Thursday AI v0.1, hit `thursday_eval_v1` ≥ 50 %.

- [ ] Scale template generator to 25k
- [ ] Build LLM-distilled generator (Method B), run on a server with API credits
- [ ] Convert Mind2Web + WebVoyager subsets to our schema
- [ ] Filter xLAM-60k subset to tool types we care about
- [ ] Mix + shuffle + dedupe → 90k JSONL
- [ ] Upload full dataset to HF Hub
- [ ] Train on Kaggle (3 epochs, ~8 hrs)
- [ ] Merge LoRA → GGUF Q4_K_M → upload to HF Hub + Ollama library
- [ ] Run `thursday_eval_v1` benchmark, log results

### Milestone M3 — Voice + runtime wiring (pre-GUI)
**Goal:** an end-to-end Python orchestrator that pipes Whisper → LLM → tools → Piper, hands-free.

- [ ] `runtime/orchestrator.py` — the think-act-observe loop
- [ ] `runtime/voice_io.py` — Whisper.cpp + Piper integration
- [ ] `runtime/tool_runtime.py` — implements all 26 tools
- [ ] `runtime/safety.py` — destructive-action confirmation
- [ ] `runtime/config.yaml` — paths, model, voice, OS profile
- [ ] End-to-end test: 10 tasks done by voice, no keyboard

### Milestone M4 — Evaluation harness
**Goal:** `thursday_eval_v1` — 50 tasks across Win/Linux/macOS, automated scoring.

- [ ] Define task spec format (goal + success predicate)
- [ ] Implement 50 tasks (alarm, app launch, browser search, file ops, screen understand, web-AI delegate)
- [ ] Implement scorer (regex / DOM check / file check / OCR check)
- [ ] Run Thursday AI v0.1 → log pass rate
- [ ] Iterate on dataset weaknesses

### Milestone M5 — GUI app (v2 kickoff)
**Goal:** standalone installer.

- [ ] Tauri shell wrapping the Python runtime
- [ ] Tray icon + settings UI
- [ ] Auto-start on boot
- [ ] Cross-platform installers

### Milestone M6 — Memory + personalisation (v2.5)
- [ ] Local-only user profile (JSON in `~/.thursday/`)
- [ ] Long-term memory: facts the user told Thursday (names, preferences)
- [ ] Short-term memory: last 10 turns kept in context
- [ ] Optional vector store (ChromaDB) for "what did I tell you about X?"

### Milestone M7 — Multimodal vision (v3)
- [ ] Plug a free VLM (Qwen2-VL-2B locally, or via `paste_to_webai` with image attachment)
- [ ] `screenshot` returns OCR + (optionally) attaches PNG to the next web-AI call

---

## 8. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 3B model too weak for multi-step agentic loops | Medium | High | Smoke test in M1 will tell us. Fallback: jump to Qwen2.5-7B + 5-bit quant on hosts that can spare 5 GB. |
| llama.cpp too slow on AVX-only i5 | Medium | High | Already chose Q4_K_M (smallest useful). If still slow, drop to Q3_K_M or move to Qwen2.5-1.5B. |
| Synthetic data overfits to teacher phrasing | High | Medium | Mix 6 sources; never let any single source > 40 % of the mixture. |
| Tool-name hallucination after fine-tune | High | Medium | Strict JSON-schema validation in the orchestrator; rejected calls become a `tool` observation of `{"error": "unknown tool"}` so the model learns to recover. |
| Kaggle session timeout mid-train | High | Low | Save LoRA adapter checkpoint every 500 steps to `/kaggle/working/`, download between sessions, resume. |
| Browser automation breaks when site UI changes | Certainty | Medium | Prefer `paste_to_webai` (headless SDK) over GUI browser when the user just wants info. Reserve GUI browser for tasks the user explicitly wants visible. |
| Wake-word detection adds latency | Medium | Medium | Use `openWakeWord` (tiny model, ~5 MB) on a separate thread; keep Whisper.cpp hot. |
| Privacy leak via browser automation | Low | High | The orchestrator must NEVER type credentials. If a site needs login, ask the user to log in once and reuse the session. |

---

## 9. Naming / Branding

- **Thursday AI** — assistant name. The user calls it "Thursday".
- **Wake word:** "Hey Thursday" (3 syllables, distinct, OpenWakeWord can learn a custom phrase).
- **Voice (TTS):** Piper's `en_US-lessac-medium` (warm, calm, ~62 MB) by default. User can swap.
- **Voice (ASR):** Whisper.cpp `small.en` (~244 MB) or `tiny.en` (~39 MB) on weaker hosts.
- **Repo structure:** see §10.

---

## 10. Repo Layout

```
Thursday_AI/
├── README.md                      ← project overview, quickstart, status
├── LIVING_PLAN.md                 ← THIS FILE — the evolving master plan
├── ARCHITECTURE.md                ← system design + data flow diagrams
├── WORKLOG.md                     ← shared agent worklog (mirror of /home/z/my-project/worklog.md)
├── LICENSE
├── research/                      ← deep research reports (Tasks 2-a/b/c)
│   ├── 01_base_model_selection.md
│   ├── 02_training_data_approach.md
│   └── 03_finetuning_recipe_kaggle.md
├── data/
│   ├── schema/
│   │   └── tools.json             ← the 26 tool definitions (JSON-schema)
│   ├── system_prompts/
│   │   └── thursday_default.md    ← the system prompt baked into every example
│   ├── templates/                 ← Jinja2 templates for Method A generation
│   │   ├── set_alarm.json
│   │   ├── open_url.json
│   │   ├── browser_search.json
│   │   ├── list_dir.json
│   │   └── ...
│   └── examples/                  ← hand-written golden JSONL examples
│       ├── alarm_7am.jsonl
│       ├── find_youtube_video.jsonl
│       ├── summarize_screen.jsonl
│       ├── delegate_to_zai.jsonl
│       └── ...
├── scripts/
│   ├── generate_template_data.py   ← Method A generator (uses data/templates/)
│   ├── generate_llm_distilled.py   ← Method B generator (uses z-ai SDK or HF Inference API)
│   ├── convert_mind2web.py         ← converts Mind2Web → our schema
│   ├── convert_webvoyager.py       ← converts WebVoyager → our schema
│   ├── filter_xlam.py              ← filters xLAM-60k to relevant tools
│   ├── validate_dataset.py         ← JSON-schema + tool-call-id consistency check
│   ├── push_to_hf.py               ← uploads dataset to HF Hub
│   └── merge_and_quantize.py       ← post-train: merge LoRA → GGUF Q4_K_M
├── training/
│   ├── thursday_ai_finetune.ipynb  ← Kaggle T4×2 notebook
│   ├── requirements.txt
│   └── README.md                  ← how to run the notebook
├── runtime/                       ← placeholder (M3 work)
│   └── README.md
└── eval/
    ├── thursday_eval_v1.json       ← 50-task benchmark spec
    └── README.md
```

---

## 11. How to Use This Plan

- **If you are the user (@skyro777):** skim §1, §4, §5, §7. The rest is detail for the build.
- **If you are a build subagent:** read §4, §5, §6, §10 to know what to produce.
- **When something changes:** edit this file, bump the revision at the top, append a row to `## Decision History`, commit with message `plan: <one-line summary>`.

---

## 12. Decision History

| Date (UTC) | Revision | Change | Why |
|---|---|---|---|
| 2025-01-01 | v0.1 | Initial plan written after research phase (Tasks 2-a, 2-b, 2-c). Locked: base model = Qwen2.5-3B-Instruct; tool vocab = 26 tools; dataset = 90k-example mix; train = Unsloth QLoRA on Kaggle T4×2. | Research showed Qwen2.5-3B is the only 3B-class model that simultaneously (a) fits 8 GB RAM at Q4_K_M, (b) has native tool_calls, (c) is Apache-2.0, (d) has Unsloth T4 notebooks ready. |

---

*End of LIVING_PLAN.md v0.1. Next update expected after M0 push and M1 smoke-train.*
