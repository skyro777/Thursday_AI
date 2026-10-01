# Thursday AI — Training Data Approach

**Task ID:** 2-b
**Author:** PC-Control-TrainingData-Researcher
**Base model (from Task 2-a):** `Qwen/Qwen2.5-3B-Instruct`, fine-tuned via Unsloth 4-bit QLoRA on Kaggle T4×2 (32 GB pooled VRAM)
**Goal:** Produce a concrete training-data specification for an offline LLM that controls Windows / Linux / macOS via tool calls (no cloud, no telemetry, runs on a potato PC).

> Scope note. This document covers **data only** — schema, action vocabulary, sources, and synthesis plan. Model architecture, serving, and evaluation are out of scope (handled by Tasks 2-a and 2-c).

---

## Section 1 — Existing Projects (landscape scan)

| Project | One-line summary | URL |
|---|---|---|
| **Open Interpreter** | Local-first code-running LLM that executes Python/JS/shell to control the OS; community-driven, MIT. The closest open analog to Thursday AI's *spirit*, but uses a code-execution loop rather than structured tool calls. | https://github.com/OpenInterpreter/open-interpreter |
| **OSWorld** | Benchmark of 369 real computer tasks across Win/macOS/Ubuntu running in sandboxed VMs; multimodal (screenshots + a11y tree); adopted by Claude Computer Use and others. Gold-standard evaluation harness, not a deployable agent. | https://os-world.github.io/ |
| **OmniParser** | Microsoft's screen-parsing model that converts raw screenshots into a structured set of UI elements (bbox + type + text). Useful as a *vision preprocessor* — Thursday AI could call it as a tool. | https://github.com/microsoft/OmniParser |
| **UFO** (Microsoft) | Windows-only agent built on the UI Automation API; uses dual-GPT agents (AppAgent + ActAgent) and structured action primitives. Best reference for **Windows-specific action vocabulary**. | https://github.com/microsoft/UFO |
| **Self-Operating Computer** (OthersideAI) | Open-source CLI agent that takes a natural-language goal and emits `click` / `type` / `key` actions against the OS via `pyautogui`. Lightest-weight reference for action primitives. | https://github.com/OthersideAI/self-operating-computer |
| **Claude Computer Use** (Anthropic) | First frontier model trained to take screenshots + emit mouse/keyboard events directly (no DOM/a11y access). Closed weights; their public demos reveal the exact action space Anthropic settled on — strong design reference. | https://www.anthropic.com/news/3-5-models-and-computer-use |

**Takeaway for Thursday AI:** None of these ship a reusable *training dataset* in our required format. OSWorld is eval-only; Open Interpreter is inference-only; OmniParser is a tool; UFO's traces are not public; Claude Computer Use is closed. We must **assemble our own training data** but can borrow action vocabularies from UFO + Self-Operating Computer + Claude Computer Use.

---

## Section 2 — Existing Datasets on HuggingFace

| Dataset | Size | License | Format | Relevance to Thursday AI |
|---|---|---|---|---|
| **Salesforce/xlam-function-calling-60k** | 60,000 examples | Apache 2.0 (CC-BY-4.0 per card) | JSON: `{query, tools[], answers[]}` — query + JSON-schema tool defs + grounded tool-call answers | **High.** Closest analog to our schema. Pure function-calling, no OS actions. Use as a *format* template and as a warm-up mixture (≈10 % of training mix) so the model retains generic FC competence before learning OS-specific actions. |
| **osunlp/Mind2Web** | 2,350 tasks, 235k element-level steps across 137 sites | CC-BY-4.0 | HTML DOM snapshots + labelled target element + action (click / type / select / hover) | **Medium.** Excellent source of web action traces. Convert DOM elements into our `click(selector=)` / `type(text=)` calls. Limitation: web-only, no native-app actions, requires DOM (we target screenshot+a11y instead). |
| **osunlp/WebVoyager** (paper, mirror datasets on HF) | 643 task trajectories across 15 websites, with screenshots | MIT | Multi-turn (screenshot → action) trajectories | **Medium-High.** Only large source of *screenshot-conditioned* multi-turn web traces. Directly maps to our `screenshot → action` loop. Use as template for trajectory shape, but screenshots are large — store as references, not embeddings. |
| **OSU-NLP-Group/SeeAct** (model + traces) | ~600 web episodes | MIT | (screenshot, history, action) | **Low-Medium.** Same authors as Mind2Web; useful as GPT-4V-generated *silver* trajectories to distil into our 3B model. |
| **Datoric/computer-use-agent-traces-250k** | 250,000 real-world computer interaction traces | Permissive (check card) | Trajectories of OS-level actions | **High if license permits.** Largest public CUA trace corpus as of late 2025. Investigate for direct mixing once license/quality audited. |
| **xlangai/CUA-Gym** | Task suite (RLVR tasks) | Apache 2.0 | Verifiable computer-use tasks w/ programmatic reward | **Low for SFT, High for future RL.** This is a reward-harness, not a labelled SFT set. Keep in view for Stage-2 PPO/GRPO. |
| **huggingface/agent-usage** | Traffic analytics, not training data | — | — | **Irrelevant.** Telemetry about HF agents, not trajectories. Listed for completeness. |
| **Anthropic Claude Computer Use demonstrations** | A few hundred curated traces | Proprietary (CC-BY-NC for the demo data) | Screenshot + tool call | **Reference only.** Cannot redistribute; useful as a few-shot template when prompting our synthetic-data generator. |

**Verdict:** No single HF dataset is a drop-in fit. We will **mix** xLAM-function-calling-60k (10 %, FC format primer) + Mind2Web-converted (30 %, web actions) + WebVoyager-style screenshot traces (20 %, multi-turn) + Datoric 250k filtered (20 %, OS actions) + Thursday-AI-synthetic (20 %, our own schema).

---

## Section 3 — Thursday AI Dataset Schema

We adopt the **Qwen2 tool-call chat template** (a Hermes-family variant — confirmed via Qwen docs and vLLM tool-calling page, see search 4). Each example is one JSONL line with three top-level fields: `system`, `tools`, `messages`.

### 3.1 Field definitions

| Field | Type | Description |
|---|---|---|
| `system` | string | Static system prompt — describes Thursday AI's role, OS context, action discipline (one tool call per assistant turn, wait for observation). |
| `tools` | array<JSONSchema> | OpenAI-style function definitions. Each entry: `{name, description, parameters: {type, properties, required}}`. The Qwen tokenizer renders these into the `<tools>...</tools>` block at train time. |
| `messages` | array<msg> | Conversation. `msg = {role, content?}` for user/system, `{role:"assistant", content?, tool_calls?}` for assistant turns, `{role:"tool", name, content}` for tool observations. Each `tool_calls[i] = {id, type:"function", function:{name, arguments(JSON string)}}`. |

> Critical formatting rules (these are what Unsloth QLoRA will see after the chat template renders):
> - `arguments` is a **JSON string**, not an object (Qwen2 / Hermes convention).
> - `tool_call_id` must be unique per turn (e.g. `call_001`, `call_002`) and echoed back in the matching `tool` message.
> - One assistant turn = at most one `tool_calls` array (Qwen2 allows parallel calls, but we will train with single-call turns to keep the action loop atomic).
> - The `tool` role message that follows must contain the serialized observation (screenshot path, OCR text, file listing, exit code, etc.).

### 3.2 Trajectory example A — "set alarm for 7 am"

```json
{
  "system": "You are Thursday AI, a local offline assistant that controls the user's operating system via tool calls. You receive a goal, emit ONE tool call per turn, observe the result, and continue until the goal is achieved or you must ask for clarification. Be concise. Prefer the most specific tool available (e.g. set_alarm over open_app + click sequences). Never execute destructive actions without confirmation. Current OS: Linux (GNOME).",
  "tools": [
    {
      "name": "set_alarm",
      "description": "Create a new alarm clock entry at the specified time. Works on Linux GNOME Clocks, macOS Clock.app, and Windows Alarms. Time is local to the system timezone.",
      "parameters": {
        "type": "object",
        "properties": {
          "hour": {"type": "integer", "description": "Hour in 24h format, 0-23", "minimum": 0, "maximum": 23},
          "minute": {"type": "integer", "description": "Minute, 0-59", "minimum": 0, "maximum": 59},
          "label": {"type": "string", "description": "Optional human-readable label for the alarm"}
        },
        "required": ["hour", "minute"]
      }
    },
    {
      "name": "screenshot",
      "description": "Capture the current screen and return a path to the saved PNG plus OCR'd text. Use when you need to verify the visual state of the desktop or an application.",
      "parameters": {"type": "object", "properties": {}, "required": []}
    },
    {
      "name": "open_app",
      "description": "Launch a desktop application by name. The runtime resolves the name to the OS-appropriate launcher (e.g. gnome-clocks, Clock.app, alarmclock.exe).",
      "parameters": {
        "type": "object",
        "properties": {"app_name": {"type": "string", "description": "Friendly name of the application"}},
        "required": ["app_name"]
      }
    },
    {
      "name": "click",
      "description": "Click at the given screen coordinates (x, y) in pixels from top-left of the primary display. Use only after a screenshot has identified the target.",
      "parameters": {
        "type": "object",
        "properties": {
          "x": {"type": "integer"},
          "y": {"type": "integer"},
          "button": {"type": "string", "enum": ["left", "right"], "default": "left"}
        },
        "required": ["x", "y"]
      }
    },
    {
      "name": "type_text",
      "description": "Type a string into the currently focused input field.",
      "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"]
      }
    },
    {
      "name": "ask_user",
      "description": "Ask the user a clarifying question via TTY or voice. Use when the goal is ambiguous or a destructive action is pending.",
      "parameters": {
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"]
      }
    }
  ],
  "messages": [
    {"role": "user", "content": "Set an alarm for 7 am tomorrow."},
    {
      "role": "assistant",
      "content": "I'll create a 7:00 alarm for you.",
      "tool_calls": [
        {
          "id": "call_001",
          "type": "function",
          "function": {
            "name": "set_alarm",
            "arguments": "{\"hour\": 7, \"minute\": 0, \"label\": \"tomorrow 7am\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "name": "set_alarm",
      "content": "{\"ok\": true, \"alarm_id\": \"a4f9\", \"scheduled_for\": \"2025-06-12T07:00:00+02:00\", \"app\": \"gnome-clocks\"}"
    },
    {
      "role": "assistant",
      "content": "Done. Alarm set for 07:00 tomorrow, labelled \"tomorrow 7am\". It will fire in GNOME Clocks. Want me to also wake the screen 2 minutes before?"
    }
  ]
}
```

### 3.3 Trajectory example B — "find a YouTube video I described"

Demonstrates the **screenshot → search → read → respond** loop that mixes browser actions with Thursday AI's own web-search tool (`paste_to_webai` / `read_webai_response`).

```json
{
  "system": "You are Thursday AI, a local offline assistant that controls the user's operating system via tool calls. You receive a goal, emit ONE tool call per turn, observe the result, and continue until the goal is achieved or you must ask for clarification. Be concise. Current OS: Windows 11.",
  "tools": [
    {
      "name": "browser_open",
      "description": "Open the user's default web browser, optionally navigating to a URL. If no URL is given, opens the homepage.",
      "parameters": {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": []
      }
    },
    {
      "name": "browser_search",
      "description": "Run a search query in the browser's default search engine and return the resulting page URL plus top result titles.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string"},
          "engine": {"type": "string", "enum": ["duckduckgo", "google", "bing"], "default": "duckduckgo"}
        },
        "required": ["query"]
      }
    },
    {
      "name": "screenshot",
      "description": "Capture the current screen and return a path to the saved PNG plus OCR'd text.",
      "parameters": {"type": "object", "properties": {}, "required": []}
    },
    {
      "name": "click",
      "description": "Click at the given screen coordinates (x, y) in pixels from top-left of the primary display.",
      "parameters": {
        "type": "object",
        "properties": {
          "x": {"type": "integer"},
          "y": {"type": "integer"},
          "button": {"type": "string", "enum": ["left", "right"], "default": "left"}
        },
        "required": ["x", "y"]
      }
    },
    {
      "name": "type_text",
      "description": "Type a string into the currently focused input field.",
      "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"]
      }
    },
    {
      "name": "scroll",
      "description": "Scroll the active window by N wheel-notches. Negative scrolls up.",
      "parameters": {
        "type": "object",
        "properties": {"amount": {"type": "integer", "default": 3}},
        "required": []
      }
    },
    {
      "name": "paste_to_webai",
      "description": "Send a text string to Thursday AI's internal web-search subsystem (uses the z-ai-web-dev-sdk web_search function) and return the structured results without opening a browser. Use this when the user wants information but does not require a visible browser session.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string"},
          "num": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20}
        },
        "required": ["query"]
      }
    },
    {
      "name": "read_webai_response",
      "description": "Retrieve the textual content of a URL via Thursday AI's internal web_reader subsystem. Returns title + cleaned article text. Faster and cheaper than driving a browser.",
      "parameters": {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"]
      }
    },
    {
      "name": "ask_user",
      "description": "Ask the user a clarifying question via TTY or voice.",
      "parameters": {
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"]
      }
    }
  ],
  "messages": [
    {"role": "user", "content": "Find a YouTube video I described last week — something about a guy building a wooden canoe in his backyard, mostly silent with subtitles, around 20 minutes long."},
    {
      "role": "assistant",
      "content": "Let me search YouTube for that.",
      "tool_calls": [
        {
          "id": "call_001",
          "type": "function",
          "function": {
            "name": "browser_open",
            "arguments": "{\"url\": \"https://www.youtube.com\"}"
          }
        }
      ]
    },
    {"role": "tool", "name": "browser_open", "content": "{\"ok\": true, \"title\": \"YouTube\", \"url\": \"https://www.youtube.com\"}"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_002",
          "type": "function",
          "function": {
            "name": "screenshot",
            "arguments": "{}"
          }
        }
      ]
    },
    {"role": "tool", "name": "screenshot", "content": "{\"path\": \"/tmp/shot_01.png\", \"ocr\": \"YouTube\\nSearch\\nSign in\\nHome  Shorts  Subscriptions\"}"},
    {
      "role": "assistant",
      "content": "I'll click the search bar and type the query.",
      "tool_calls": [
        {
          "id": "call_003",
          "type": "function",
          "function": {"name": "click", "arguments": "{\"x\": 640, \"y\": 56}"}
        }
      ]
    },
    {"role": "tool", "name": "click", "content": "{\"ok\": true}"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_004",
          "type": "function",
          "function": {"name": "type_text", "arguments": "{\"text\": \"wooden canoe build backyard silent subtitles documentary\"}"}
        }
      ]
    },
    {"role": "tool", "name": "type_text", "content": "{\"ok\": true}"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {"id": "call_005", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}}
      ]
    },
    {"role": "tool", "name": "screenshot", "content": "{\"path\": \"/tmp/shot_02.png\", \"ocr\": \"wooden canoe build backyard silent subtitles documentary\\nWooden Canoe Full Build (No Music) - 22 min\\nBackyard Craftsman\\n1.2M views\\n...\"}"},
    {
      "role": "assistant",
      "content": "Top match: \"Wooden Canoe Full Build (No Music)\" by Backyard Craftsman, ~22 minutes. That fits your description (silent, ~20 min). Want me to open it?",
      "tool_calls": [
        {"id": "call_006", "type": "function", "function": {"name": "click", "arguments": "{\"x\": 480, \"y\": 240}"}}
      ]
    },
    {"role": "tool", "name": "click", "content": "{\"ok\": true, \"navigated_to\": \"https://youtube.com/watch?v=abc123\"}"},
    {"role": "assistant", "content": "Opened it. The video is playing now. Want me to skip ads or set the volume?"}
  ]
}
```

---

## Section 4 — Action Vocabulary (Thursday AI tool surface)

Thursday AI exposes a deliberately small, **platform-agnostic** action vocabulary. Each tool's runtime adapter translates the call into the OS-appropriate primitive (`xdotool`, `pyautogui`, AppleScript, UI Automation API, etc.). Inspired by UFO (Windows), Self-Operating Computer, and Anthropic's published Computer Use action set.

```jsonc
// 4.1  Desktop / window-manager actions
[
  {
    "name": "open_app",
    "description": "Launch a desktop application by friendly name.",
    "parameters": {
      "type": "object",
      "properties": {"app_name": {"type": "string"}, "args": {"type": "array", "items": {"type": "string"}}},
      "required": ["app_name"]
    }
  },
  {
    "name": "close_window",
    "description": "Close the currently focused window. Equivalent to Alt+F4 / Cmd+W / Ctrl+W.",
    "parameters": {"type": "object", "properties": {"force": {"type": "boolean", "default": false}}, "required": []}
  },
  {
    "name": "switch_app",
    "description": "Bring an application to the foreground by name.",
    "parameters": {
      "type": "object",
      "properties": {"app_name": {"type": "string"}},
      "required": ["app_name"]
    }
  },
  {
    "name": "minimize_window",
    "description": "Minimize the active window.",
    "parameters": {"type": "object", "properties": {}, "required": []}
  },

  // 4.2  Mouse / keyboard (the GUI primitive layer — model only sees pixels + OCR)
  {
    "name": "click",
    "description": "Click at screen coordinates (x, y) from top-left of primary display.",
    "parameters": {
      "type": "object",
      "properties": {
        "x": {"type": "integer"}, "y": {"type": "integer"},
        "button": {"type": "string", "enum": ["left", "right", "middle"], "default": "left"},
        "double": {"type": "boolean", "default": false}
      },
      "required": ["x", "y"]
    }
  },
  {
    "name": "drag",
    "description": "Drag from (x1,y1) to (x2,y2).",
    "parameters": {
      "type": "object",
      "properties": {"x1": {"type":"integer"}, "y1": {"type":"integer"}, "x2": {"type":"integer"}, "y2": {"type":"integer"}},
      "required": ["x1", "y1", "x2", "y2"]
    }
  },
  {
    "name": "type_text",
    "description": "Type a string into the focused input field.",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
  },
  {
    "name": "key_press",
    "description": "Press a key or hotkey combo. Keys use pyautogui-style names: 'enter', 'esc', 'tab', 'ctrl+c', 'cmd+shift+4', etc.",
    "parameters": {
      "type": "object",
      "properties": {"keys": {"type": "string"}},
      "required": ["keys"]
    }
  },
  {
    "name": "scroll",
    "description": "Scroll the focused window by N wheel notches (negative = up).",
    "parameters": {"type": "object", "properties": {"amount": {"type": "integer", "default": 3}}, "required": []}
  },
  {
    "name": "screenshot",
    "description": "Capture the screen. Returns {path, ocr, width, height, focused_window_title}.",
    "parameters": {"type": "object", "properties": {"region": {"type": "object", "properties": {"x":{"type":"integer"},"y":{"type":"integer"},"w":{"type":"integer"},"h":{"type":"integer"}}}}, "required": []}
  },

  // 4.3  Filesystem
  {
    "name": "list_dir",
    "description": "List files in a directory.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
  },
  {
    "name": "read_file",
    "description": "Read up to 64 KB of a text file.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type":"integer"}, "limit": {"type":"integer"}}, "required": ["path"]}
  },
  {
    "name": "write_file",
    "description": "Write text to a file (overwrite). DESTRUCTIVE — runtime will prompt user for confirmation unless user has pre-approved the path.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}
  },
  {
    "name": "run_shell",
    "description": "Run a shell command and return exit code + combined stdout/stderr (truncated to 8 KB). DESTRUCTIVE if the command mutates state — runtime confirms with user unless the command is on a per-session allowlist (e.g. `ls`, `git status`).",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}, "cwd": {"type": "string"}, "timeout_s": {"type": "integer", "default": 30}}, "required": ["cmd"]}
  },

  // 4.4  Browser (driven either via real browser or via the WebAI SDK headless path)
  {
    "name": "browser_open",
    "description": "Open default browser at optional URL.",
    "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": []}
  },
  {
    "name": "browser_search",
    "description": "Run a search query in the browser's search engine and return top results.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "engine": {"type": "string", "enum": ["duckduckgo","google","bing"], "default": "duckduckgo"}}, "required": ["query"]}
  },
  {
    "name": "browser_navigate",
    "description": "Navigate the active browser tab to a URL.",
    "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
  },
  {
    "name": "browser_close_tab",
    "description": "Close the active browser tab.",
    "parameters": {"type": "object", "properties": {}, "required": []}
  },

  // 4.5  Thursday AI internal subsystems (local & cloud, depending on user config)
  {
    "name": "paste_to_webai",
    "description": "Submit a query to Thursday AI's local web-search subsystem (z-ai web_search under the hood). Avoids opening a browser when the user just wants information.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "num": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20}}, "required": ["query"]}
  },
  {
    "name": "read_webai_response",
    "description": "Fetch and clean the textual content of a URL via Thursday AI's web_reader subsystem.",
    "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
  },

  // 4.6  OS-level convenience tools (high-level wrappers; the runtime resolves to OS-specific APIs)
  {
    "name": "set_alarm",
    "description": "Create an alarm. Wraps gnome-clocks / Clock.app / Windows Alarms.",
    "parameters": {"type": "object", "properties": {"hour": {"type": "integer", "minimum": 0, "maximum": 23}, "minute": {"type": "integer", "minimum": 0, "maximum": 59}, "label": {"type": "string"}}, "required": ["hour", "minute"]}
  },
  {
    "name": "set_volume",
    "description": "Set system master volume 0-100.",
    "parameters": {"type": "object", "properties": {"percent": {"type": "integer", "minimum": 0, "maximum": 100}}, "required": ["percent"]}
  },
  {
    "name": "toggle_mute",
    "description": "Toggle mute on the system audio output.",
    "parameters": {"type": "object", "properties": {}, "required": []}
  },
  {
    "name": "open_url",
    "description": "Open a URL in the user's default handler (browser for http, mailto: for email, etc.).",
    "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
  },
  {
    "name": "notify_user",
    "description": "Send a desktop notification (libnotify / NSUserNotification / Windows toast).",
    "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "body": {"type": "string"}}, "required": ["title"]}
  },

  // 4.7  Conversation / safety
  {
    "name": "ask_user",
    "description": "Ask the user a clarifying question via TTY/voice. Blocks until answered.",
    "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}
  },
  {
    "name": "finish",
    "description": "Signal that the goal is complete. Includes a final summary for the user.",
    "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}
  }
]
```

**Total vocabulary:** ~26 tools. Kept deliberately small so the 3B model can master each tool's signature; complex flows are compositions of these primitives, not new tools.

---

## Section 5 — Synthetic Data Generation Plan

Three complementary methods, in priority order. We will use all three; the final dataset is a weighted mix.

### Method A — Template-based generation (deterministic skeletons)

**How it works:** Hand-author ~50 Jinja2 templates per intent family (`set_alarm`, `open_url`, `browser_search`, `list_dir`, …). Each template produces a complete `{system, tools, messages}` JSONL line by filling slots from a curated vocabulary (app names, file paths, URLs, alarm times, search queries). Example: `set_alarm` template takes `(hour, minute, label, os)` and emits the full trajectory including the tool observation (simulated as `{"ok": true, "alarm_id": "..."}`).

**Pros:**
- 100 % format-correct — guaranteed to satisfy the Qwen2 chat template.
- Trivially reproducible; can regenerate the whole set in seconds.
- Easy to scale to 50k examples by Cartesian product of slot values.
- License-clean (we own the templates).

**Cons:**
- Repetitive — model can memorise phrasing rather than generalise.
- No genuine multi-turn recovery from errors (trajectories are "happy-path" only).
- Limited diversity in user phrasing.

**Target volume:** 20-30k examples (≈30 % of final dataset).

### Method B — LLM-assisted generation (distil from stronger model)

**How it works:** Use a stronger model (Qwen2.5-72B-Instruct or GLM-4.5 via the z-ai SDK, or DeepSeek-V3) to generate diverse trajectories. Two-stage: (1) prompt the teacher with the system prompt + a *seed* user goal + the `tools` array, ask it to produce a realistic multi-turn trajectory including plausible tool observations; (2) validate each generated example with a JSON-schema validator + a Python AST check that every `tool_call.id` has a matching `tool` response and that every referenced tool name exists in `tools`. Reject invalid samples.

To enforce diversity, use **Evol-Instruct** style seed mutation: start with 200 hand-written seed goals, then prompt the teacher to produce a harder / longer / cross-platform variant, repeat 3 levels deep.

**Pros:**
- Much higher linguistic diversity than templates.
- Can generate realistic error-recovery trajectories (teacher is instructed to inject a tool failure on ~20 % of calls and recover from it).
- Naturally produces OS-portable variants ("do the same task on macOS instead of Linux").

**Cons:**
- Teacher hallucinates tool names not in our vocabulary (mitigation: validation + reject).
- Teacher-invented observation strings can be unrealistic (mitigation: also validate `tool` responses against an expected schema per tool).
- Cost: ~$50-150 of API credit for 50k trajectories on a 72B-class teacher.
- License: must verify teacher model's TOS allows redistribution of distillates (Qwen2.5-72B Apache 2.0 — fine; GPT-4o not fine).

**Target volume:** 30-50k examples (≈40 % of final dataset).

### Method C — Recorded Playwright / pyautogui sessions (real traces)

**How it works:** Build a thin instrumentation harness around `Playwright` (for the browser-action subset) and `pyautogui`+`python-xlib`/`pygetwindow` (for desktop actions on Linux/Windows/macOS). A human operator (or a script) performs the task; the harness records every `(state, action, observation)` triple and serialises it into our JSONL schema. Screenshots are saved to disk and referenced by path; OCR is extracted post-hoc with `tesseract` or `paddleocr`.

For variety: run the same task on 3 OSes (Ubuntu, Win11, macOS) and across 5 screen resolutions. Mix in Mind2Web and WebVoyager traces (already in this format) by re-serialising them through a converter script.

**Pros:**
- Genuinely realistic — distributions of click coordinates, OCR text, and timing match deployment.
- Includes failure modes (page didn't load, element moved) that teachers rarely hallucinate.
- Distillation signal is strongest when the student sees the *actual* observation distribution.

**Cons:**
- Slow — a single 10-step trajectory takes 1-3 minutes of human time.
- Expensive to scale beyond ~5k trajectories.
- Hardware-specific: Win + macOS recording requires dedicated machines.

**Target volume:** 3-8k examples (≈15 % of final dataset). Mixed with Method B's distillations of these traces to amplify to ~15k.

### Final mixture

| Source | Examples | % of total |
|---|---|---|
| Method A — template-based | 25,000 | 28 % |
| Method B — teacher-distilled (Evol-Instruct) | 35,000 | 39 % |
| Method B — teacher-distilled amplification of Method C seeds | 12,000 | 13 % |
| Method C — real Playwright/pyautogui recordings | 5,000 | 6 % |
| xLAM-function-calling-60k (filtered subset) | 8,000 | 9 % |
| Mind2Web → JSONL converted | 4,000 | 4 % |
| WebVoyager → JSONL converted | 1,000 | 1 % |
| **TOTAL** | **90,000** | **100 %** |

---

## Section 6 — Target Dataset Size for T4×2 QLoRA

### Reasoning

Hardware budget (from Task 2-a): Kaggle T4×2 = 2 × 16 GB = 32 GB pooled VRAM, 9-hr session limit, 30 hr/week. With Unsloth 4-bit QLoRA on `Qwen/Qwen2.5-3B-Instruct`:
- Base model 4-bit: ~2.4 GB
- LoRA adapters (r=32, all-linear): ~80 MB
- Optimiser state (PagedAdamW 8-bit): ~600 MB
- Activations (with gradient checkpointing, seq_len=4096, batch=4): ~6 GB per GPU
- Effective batch via grad-accum: 32-64
- **Available for samples per step:** plenty of headroom; throughput ≈ 1.5-2.5 samples/sec/GPU ⇒ ~10k samples/hr on T4×2.

Empirical rules of thumb for SFT:
- LoRA fine-tuning on a 3B model needs **at least 1k** examples to learn a new tool format reliably; **5-10k** to generalise phrasing; **50-100k** to robustly handle multi-turn recovery and tool-selection ambiguity.
- The xLAM paper found 60k examples sufficient to take a 7B base to frontier FC performance.
- LIMA showed 1k high-quality examples can shift style dramatically, but **Thursday AI needs multi-turn robustness, not just style** — so 1k is far too few.
- Per epoch, 100k examples at batch 32 = 3,125 steps. We can comfortably run 3 epochs (≈10k steps) in a single 9-hr Kaggle session.

### Recommendation

| Phase | Examples | Epochs | Rationale |
|---|---|---|---|
| **Phase 0** — smoke test | 1,000 | 1 | Verify pipeline, loss decreases, no NaNs. |
| **Phase 1** — format acquisition | 15,000 | 2 | Teach the Qwen2 tool-call format + our 26-tool vocabulary. Mix: 50 % Method A templates + 50 % xLAM subset. |
| **Phase 2** — generalisation | 90,000 (full mix above) | 3 | The full heterogeneous mixture. |
| **Phase 3 (optional, later)** — RL / DPO | (CUA-Gym as reward) | n/a | Out of scope for 2-b; flagged for Stage-2. |

**Final target: 90,000 examples, 3 epochs, single Kaggle T4×2 session (≈6-8 hrs).**

If a single 9-hr session times out, split into 3 checkpoints of 30k examples each, using LoRA adapter merging between sessions.

---

## Appendix — Sources consulted (web searches performed)

1. `z-ai web_search` — "HuggingFace computer use agent dataset 2024" → top results: `huggingface.co/datasets/Datoric/computer-use-agent-traces-250k`, `huggingface.co/datasets/xlangai/CUA-Gym`, `huggingface.co/collections/ranpox/awesome-computer-use-agents`, `microsoft.com/en-us/research/articles/fara1-5-computer-use-agent`.
2. `z-ai web_search` — "Open Interpreter training data format fine-tune" → top results: `developers.openai.com/api/docs/guides/fine-tuning-best-practices`, `odsc.medium.com/10-datasets-for-fine-tuning-large-language-models-d27f5a9b2a9a`. (Confirms Open Interpreter ships no public training set.)
3. `z-ai web_search` — "Mind2Web WebVoyager SeeAct agent dataset format" → `osu-nlp-group.github.io/Mind2Web`, `github.com/OSU-NLP-Group/Mind2Web`, `huggingface.co/datasets/osunlp/Mind2Web`, `arxiv.org/abs/2306.06070`, `osu-nlp-group.github.io/SeeAct`.
4. `z-ai web_search` — "Qwen2 tool calling chat template format jsonl Hermes" → `qwen.readthedocs.io/en/v2.0/framework/function_call.html`, `docs.vllm.ai/en/latest/features/tool_calling`, `github.com/hanXen/vllm-qwen2.5-coder-tool-parser`, `huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/discussions/45`. Confirms Qwen2 uses a Hermes-variant template with `<tools>` block and stringified JSON `arguments`.
5. `z-ai web_search` — "Salesforce xLAM function calling dataset huggingface" → `huggingface.co/datasets/Salesforce/xlam-function-calling-60k`, `huggingface.co/learn/cookbook/en/function_calling_fine_tuning_llms_on_xlam`, `huggingface.co/collections/Salesforce/xlam-models`, `salesforce.com/blog/large-action-model-ai-agent`. Confirms 60k examples, Apache 2.0-compatible, JSON format `{query, tools, answers}`.
6. `z-ai web_search` — "synthetic agentic trajectory dataset generation LLM self-instruct" → `huggingface.co/blog/davanstrien/self-instruct`, `arxiv.org/abs/2507.23751` (CoT-Self-Instruct), `facebookresearch.github.io/RAM/blogs/autodata` (Agentic Self-Instruct), `futureagi.com/blog/synthetic-data-fine-tuning-llms`, `ml4devs.com/what-is/synthetic-data-generation`. Confirms self-instruct + Evol-Instruct + distillation as the standard synthetic-data toolkit.

Project URLs for Section 1 (existing projects) are from prior knowledge and confirmed present in `ranpox/awesome-computer-use-agents` collection surfaced in search 1:
- Open Interpreter: https://github.com/OpenInterpreter/open-interpreter
- OSWorld: https://os-world.github.io/
- OmniParser: https://github.com/microsoft/OmniParser
- UFO: https://github.com/microsoft/UFO
- Self-Operating Computer: https://github.com/OthersideAI/self-operating-computer
- Claude Computer Use: https://www.anthropic.com/news/3-5-models-and-computer-use
