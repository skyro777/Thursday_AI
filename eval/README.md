# Eval — Thursday AI (placeholder, M4 work)

This folder holds the **`thursday_eval_v1`** benchmark — 50 cross-OS tasks (alarm, app launch, browser search, file ops, screen understanding, web-AI delegation, composition) used to score Thursday AI releases.

**Status:** 8-task seed written in `thursday_eval_v1.json`. Full 50-task benchmark + scorer tracked as Milestone M4 in `LIVING_PLAN.md`.

## Task format

```json
{
  "id": "eval_001",
  "category": "alarm",
  "goal": "Set an alarm for 7 am tomorrow.",
  "success_predicate": "An alarm exists in the OS clock app set for 07:00 the following day.",
  "max_steps": 4,
  "os": ["linux", "windows", "macos"]
}
```

The scorer (to be implemented) will:
1. Boot a sandboxed VM (Linux / Windows / macOS)
2. Start Thursday AI + the orchestrator
3. Speak the `goal` (via simulated ASR input)
4. Let Thursday AI run up to `max_steps` tool calls
5. Check the `success_predicate` via DOM inspection / file check / OCR / clock-app API

Pass / fail logged per task. Aggregate pass-rate is the headline metric.

## Target

- v0.1 release: ≥ 50 % pass rate
- v1.0 release: ≥ 80 % pass rate
