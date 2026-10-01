# Thursday AI — Architecture

> Companion to `LIVING_PLAN.md`. This file describes **how the pieces fit together** at runtime and at training time. It is intentionally short — the plan is in `LIVING_PLAN.md`, the deep research is in `research/`, the code is in `scripts/` and `training/`.

---

## 1. Two Modes of the Same System

Thursday AI has two operating modes that share the same brain:

```
        ┌──────────────────────── TRAINING (offline, Kaggle) ────────────────────────┐
        │                                                                            │
        │   data/schema/tools.json ──┐                                               │
        │   data/examples/*.jsonl ───┼──► 90k JSONL ──► Unsloth QLoRA ──► LoRA adapter
        │   data/templates/*.json ───┤       (Qwen2.5-3B-Instruct, 4-bit, T4×2)        │
        │   external datasets ───────┘                            │
        │                                                         ▼
        │                                          merge LoRA → GGUF Q4_K_M
        │                                                         │
        │                                                         ▼
        │                                          thursday-ai-v0.1.gguf  (1.93 GB)
        └─────────────────────────────────────────────────────────┬─────────────────┘
                                                                  │
                                                                  ▼
        ┌──────────────────────── INFERENCE (offline, user PC) ───────────────────┐
        │                                                                         │
        │   microphone ──► Whisper.cpp ──► text                                  │
        │                                    │                                    │
        │                                    ▼                                    │
        │   ┌─────────────────────────────────────────────────┐                    │
        │   │  Thursday Orchestrator (Python)                 │                    │
        │   │  • holds system prompt + tool schemas           │                    │
        │   │  • calls llama.cpp / Ollama HTTP API            │                    │
        │   │  • parses tool_calls, dispatches to runtime      │                    │
        │   │  • enforces safety (asks user before destructive)│                    │
        │   │  • limits reply length (3-7 words while working)│                    │
        │   └──────────┬──────────────────────────┬──────────┘                    │
        │              │ tool_call                 │ observation                   │
        │              ▼                           │                               │
        │   ┌────────────────────────────┐         │                               │
        │   │  Tool Runtime (Python)     │◄────────┘                               │
        │   │  • pyautogui (mouse/kb)    │                                         │
        │   │  • playwright (browser)    │                                         │
        │   │  • subprocess (shell)      │                                         │
        │   │  • mss (screenshots)        │                                        │
        │   │  • tesseract (OCR)          │                                         │
        │   │  • z-ai-web-dev-sdk          │                                        │
        │   │    (web_search, web_reader) │                                         │
        │   │  • platform-specific utils  │                                         │
        │   │    (AppleScript / xdotool / │                                         │
        │   │     Win UI Automation)      │                                         │
        │   └────────────┬───────────────┘                                         │
        │                │ final text                                               │
        │                ▼                                                          │
        │   Piper TTS ──► speaker                                                   │
        └─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Think-Act-Observe Loop

This is the heart of the runtime. Every user request triggers it.

```
1.  user_utterance = ASR(audio)                       # Whisper.cpp
2.  history.append({role: "user", content: user_utterance})
3.  loop:
4.      response = llama_complete(                    # llama.cpp / Ollama
5.          model    = thursday-ai-v0.1.gguf,
6.          messages = [system] + history + tool_schemas,
7.          tools    = tools.json,
8.          options  = {temperature: 0.3, max_tokens: 200}
9.      )
10.     history.append(response.message)
11.
12.     if response.message.tool_calls is empty:
13.         # final answer
14.         speak(response.message.content)            # Piper TTS
15.         break
16.
17.     for call in response.message.tool_calls:
18.         if is_destructive(call) and not user_approved(call):
19.             obs = {"error": "user declined"}
20.         else:
21.             obs = tool_runtime.dispatch(call)
22.         history.append({role:"tool", name:call.name, content: json(obs)})
23.         speak(short_status(call, obs))             # "Got it." / "Done." / "Opening..."
24.     # loop continues — model decides whether to call another tool or finish
```

**Key invariants:**
- The model **always** sees the full `tools.json` array in every request (Qwen2 renders it into the `<tools>` block automatically).
- The model emits **at most one tool call per turn** (we train it that way; the orchestrator enforces it by truncating extra calls).
- Tool observations are JSON-serialised strings, never raw Python objects.
- The orchestrator speaks a **≤ 7-word** status after every tool execution. The model itself never writes these — they come from a tiny lookup table keyed by `(tool_name, success|fail)`. This guarantees the "short while working" behavior regardless of model verbosity.

---

## 3. Tool Call Lifecycle

```
Model emits:  {"name":"click","arguments":"{\"x\":640,\"y\":56}"}
        │
        ▼
Orchestrator validates against data/schema/tools.json
        │  ✓ valid
        ▼
Orchestrator checks safety rules:
        │  • click on non-destructive target?  → no confirmation needed
        │  • click on file-deleting button?    → ask user (voice)
        ▼
Tool Runtime dispatches:
        │  • Windows:  pyautogui.click(640, 56)
        │  • Linux:    xdotool mousemove 640 56 click 1
        │  • macOS:    pyautogui.click(640, 56)  (or AppleScript)
        ▼
Observation captured:
        │  {"ok": true, "x": 640, "y": 56, "focused_window": "YouTube - Mozilla Firefox"}
        ▼
JSON-serialised → appended to history as {role:"tool", name:"click", content:...}
        │
        ▼
Orchestrator speaks short status ("Got it.") → loops back to model
```

---

## 4. The Web-AI Delegation Pattern

One of Thursday AI's signature moves: **delegate to a heavier free web AI** when the local 3B model isn't smart enough. Two flavors:

### 4a. Headless (fast, invisible)
```python
# Model calls:
paste_to_webai(query="write a python script that downloads every image from a webpage",
               context="user wants a reusable script, prefer requests+bs4")
# Tool runtime:
result = z_ai_web_dev_sdk.chat.completions.create(
    model="glm-4.5",
    messages=[{"role":"user","content": query + "\n\nContext: " + context}],
    max_tokens=2000
)
# Observation returned to model:
{"answer": "<the script>", "model":"glm-4.5", "tokens_used": 487}
```

### 4b. Visible (user sees the browser being driven)
For tasks where the user explicitly wants to watch (e.g., "open Grok and ask it to summarize this video"):
```python
# Model calls:
browser_open(url="https://grok.com")
screenshot()                                     # verify page loaded
click(x=400, y=120)                              # click the chat input
type_text(text="<the prompt>")                  # type the question
key_press(keys="ctrl+enter")                     # send
sleep(5)                                         # let Grok think
screenshot()                                     # capture response
# OCR'd text becomes the observation; model reads it and replies to user
```

The choice between (a) and (b) is something the model learns from training data — examples of both are in `data/examples/`.

---

## 5. Safety Architecture

Three layers:

1. **Schema validation** — every tool call is validated against `data/schema/tools.json`. Invalid calls → observation `{"error":"schema_invalid", "detail":"..."}`. The model recovers.

2. **Destructive-action gate** — `write_file`, `run_shell`, `close_window` (with `force=true`), `browser_navigate` to URLs matching `(login|bank|payment|admin)` patterns → orchestrator intercepts, speaks to the user via TTS: *"About to run `rm -rf ~/.cache`. Say 'yes' to confirm."* — waits for wake-word-channelled yes/no.

3. **Audit log** — every tool call + observation + final reply is appended to `~/.thursday/audit_YYYYMMDD.jsonl`. User can review. Never sent anywhere.

---

## 6. Memory (M6, not in v0.1)

For v1, history is per-session: cleared on restart. M6 adds:
- `~/.thursday/profile.json` — long-term facts ("user's name is X", "user prefers dark mode")
- `~/.thursday/memory/` — ChromaDB vector store for "what did I tell you about X?"
- Both are local files; never uploaded.

---

## 7. Inference Engine Choices

| Engine | Use when | Pros | Cons |
|---|---|---|---|
| **llama.cpp (built from source, AVX1, no AVX2)** | Production on potato PC | ~30 % faster than Ollama; lowest latency | Build complexity |
| **Ollama** | Easy install / dev | Auto noavx fallback on Windows; one-line model pull | Slower; HTTP overhead |
| **KoboldCpp** | Fallback if both above fail | Bundles noavx build; good UI | Heavier |

Default ship: **Ollama for installation simplicity; llama.cpp as the recommended performance upgrade**.

---

## 8. Training Pipeline (detailed flow)

```
data/schema/tools.json
data/system_prompts/thursday_default.md
data/templates/*.json  ─────► generate_template_data.py ────┐
data/examples/*.jsonl  ─────────────────────────────────────┤
                                                              ├─► 90k JSONL
[HF Hub: xLAM-60k] ────► filter_xlam.py ─────────────────────┤
[HF Hub: Mind2Web] ────► convert_mind2web.py ────────────────┤
[HF Hub: WebVoyager] ─► convert_webvoyager.py ──────────────┤
                                                              │
                                                              ▼
                                                validate_dataset.py  (JSON-schema + tool-call-id check)
                                                              │
                                                              ▼
                                          training/thursday_ai_finetune.ipynb  (Kaggle T4×2)
                                                              │
                                                              ▼
                                          LoRA adapter  ──►  merge_and_quantize.py
                                                              │
                                                              ▼
                                          thursday-ai-v0.1-Q4_K_M.gguf  (1.93 GB)
                                                              │
                                                              ▼
                                          Push to HF Hub  +  Ollama library
```

---

## 9. Performance Budget (potato PC)

Target: i5-3rd-gen, 8 GB DDR3, Intel HD 4000, no dGPU.

| Stage | Target latency | Budget |
|---|---|---|
| Wake word detection (openWakeWord) | 100 ms | always-on, ~50 MB RAM |
| ASR (Whisper tiny.en, 39 MB) | 300-500 ms for 2-3 s utterance | loaded hot |
| LLM (Qwen2.5-3B Q4_K_M, llama.cpp) | 1.0-1.5 s for 7-word reply (~10 tokens @ 7 tok/s) | main consumer; ~2.4 GB RAM |
| Tool execution (single tool, e.g., click) | 50-200 ms | trivial |
| TTS (Piper lessac-medium, 62 MB) | 200-400 ms for 7 words | loaded hot |
| **End-to-end reply** | **< 3 s** | acceptable for voice |

If this is too slow in M3, drop to `Qwen2.5-1.5B-Instruct` (Q4_K_M = 0.9 GB, ~15 tok/s) — quality hit but latency halved.

---

## 10. Failure Modes We Design For

| Failure | What happens | Recovery |
|---|---|---|
| Tool call references unknown tool | Orchestrator returns `{"error":"unknown_tool"}` | Model retrains itself in-context to use only known tools |
| Tool call has bad arguments | Orchestrator returns `{"error":"schema_invalid", "detail":"..."}` | Model re-issues with corrected args |
| Screenshot returns blank / OCR fails | Orchestrator returns `{"error":"screen_unreadable"}` + asks user to describe | Model falls back to asking the user |
| Browser automation hangs | 30-s timeout → `{"error":"timeout"}` | Model retries once, then asks user |
| Web-AI delegation returns garbage | Orchestrator surfaces raw response | Model decides whether to retry or ask user |
| LLM emits refusal / safety lecture | Orchestrator detects "I can't" patterns → strips them, appends "What would you do if you could?" system nudge | Model recovers |

---

*End of ARCHITECTURE.md. Update when the system shape changes.*
