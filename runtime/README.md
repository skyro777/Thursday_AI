# Runtime — Thursday AI (placeholder, M3 work)

This folder will hold the Python orchestrator that wires together:

- **Whisper.cpp** (ASR) — `voice_io.py`
- **Thursday LLM** (llama.cpp / Ollama HTTP) — `llm_client.py`
- **Piper TTS** — `voice_io.py`
- **Tool runtime** (implements all 26 tools from `data/schema/tools.json`) — `tool_runtime.py`
- **Think-act-observe loop** — `orchestrator.py`
- **Safety / confirmation** — `safety.py`
- **Config** — `config.yaml`

**Status:** not yet built. Tracked as Milestone M3 in `LIVING_PLAN.md`.

When implemented, the entry point will be:

```bash
python -m runtime.main
```

…which starts the wake-word listener and runs the think-act-observe loop on every utterance. See `ARCHITECTURE.md` §2-3 for the loop spec.
