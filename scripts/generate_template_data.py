#!/usr/bin/env python3
"""
generate_template_data.py — Thursday AI Method A dataset generator.

Produces thousands of JSONL training examples from hand-authored trajectory
templates. Each template is a Python function that yields one example dict
given a set of slot values. We cross-product the slot values to scale.

Output: data/processed/template_generated.jsonl

Usage:
    python scripts/generate_template_data.py --count 1000 --out data/processed/template_generated.jsonl
    python scripts/generate_template_data.py --count 5000 --seed 42
    python scripts/generate_template_data.py --list-templates
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "data" / "schema" / "tools.json"
OUTPUT_DEFAULT = REPO_ROOT / "data" / "processed" / "template_generated.jsonl"

# ---------------------------------------------------------------------------
# Shared slot vocabularies (kept small but diverse enough to avoid memorisation)
# ---------------------------------------------------------------------------

USER_NAMES = ["Alex", "Sam", "Jordan", "Riya", "Diego", "Mei", "Noah", "Priya", "Liam", "Zoe"]

OS_PROFILES = [
    "Linux (GNOME)",
    "Linux (KDE Plasma)",
    "Windows 11",
    "Windows 10",
    "macOS 14",
    "macOS 13",
]

# Map (os_profile, app_friendly_name) -> resolution; we don't actually need the
# resolution in training data (the runtime handles it), but we encode the
# friendly name in the tool call args.

APPS_BY_OS = {
    "Linux (GNOME)": ["firefox", "chromium", "vscode", "gnome-clocks", "files", "spotify", "terminal", "gnome-text-editor", "thunderbird", "blender"],
    "Linux (KDE Plasma)": ["firefox", "chromium", "vscode", "konsole", "dolphin", "spotify", "kate", "kmail", "gimp"],
    "Windows 11": ["firefox", "chrome", "vscode", "notepad", "explorer", "spotify", "powershell", "clocks", "thunderbird", "blender"],
    "Windows 10": ["firefox", "chrome", "vscode", "notepad", "explorer", "spotify", "cmd", "clocks", "thunderbird"],
    "macOS 14": ["safari", "firefox", "vscode", "notes", "finder", "spotify", "terminal", "clocks", "mail", "blender"],
    "macOS 13": ["safari", "firefox", "vscode", "notes", "finder", "spotify", "terminal", "clocks", "mail"],
}

SEARCH_ENGINES = ["duckduckgo", "google", "bing", "brave"]
ALARM_LABELS = ["wake up", "standup meeting", "take meds", "leave for flight", "call mom", "lunch", "gym", "study break", "tomorrow 7am", "laundry"]
VOLUME_LEVELS = [(20, "low"), (40, "background"), (60, "normal"), (80, "loud"), (100, "max")]
NEWS_QUERIES = ["top world news today", "tech news this week", "AI news today", "climate news headlines", "stock market today", "sports scores today"]
FILE_PATHS = ["~/Downloads/notes.txt", "~/Documents/todo.md", "~/projects/main.py", "~/Pictures/vacation.txt", "~/Desktop/scratch.txt", "~/Music/playlist.txt"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tools() -> list[dict]:
    """Load the full tool surface. Templates can either pass the full 26-tool
    array (so the model sees every tool even if a turn only uses one), or pass
    a subset (so the model learns to ignore irrelevant tools).
    For training, we always pass the full 26-tool array for realism."""
    with SCHEMA_PATH.open() as f:
        return json.load(f)["tools"]


def system_prompt(os_profile: str, now_iso: str, user_name: str) -> str:
    return (
        "You are Thursday, a local offline AI assistant that controls the user's operating "
        "system through tool calls. You run on the user's own PC; nothing leaves their machine "
        "unless you explicitly open a browser or call paste_to_webai on their behalf.\n\n"
        "BEHAVIOR:\n"
        "- Receive the user's goal, think briefly, then emit exactly ONE tool call per turn. "
        "Wait for the observation before continuing.\n"
        "- While a task is in progress, your spoken status replies must be 3-7 words.\n"
        "- When the goal is complete, call finish() with a one-or-two-line summary.\n"
        "- Prefer the most specific tool available.\n"
        "- Take a screenshot before clicking or typing into a UI you cannot currently see.\n\n"
        "DELEGATION:\n"
        "- If a task needs more intelligence than you have, delegate: call paste_to_webai with a "
        "well-crafted prompt and use the response.\n\n"
        "SAFETY & PRIVACY:\n"
        "- Never type the user's passwords, tokens, or credentials.\n"
        "- Never send the user's personal files anywhere.\n"
        "- If unsure whether an action is safe, ask_user.\n\n"
        f"CURRENT OS: {os_profile}\n"
        f"CURRENT TIME: {now_iso}\n"
        f"USER NAME: {user_name}"
    )


def call(call_id: str, name: str, args: dict) -> dict:
    """Build a Qwen2-style tool_call object."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def obs(name: str, payload: dict) -> dict:
    """Build a tool observation message."""
    return {"role": "tool", "name": name, "content": json.dumps(payload, ensure_ascii=False)}


def assistant(content: str | None, tool_calls: list[dict] | None = None) -> dict:
    msg: dict[str, Any] = {"role": "assistant"}
    if content is not None:
        msg["content"] = content
    else:
        msg["content"] = None
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def random_iso_time(rng: random.Random) -> str:
    """Random ISO timestamp within the last 30 days, localised to a plausible tz."""
    base = datetime(2025, 6, 12, 22, 14, 0, tzinfo=timezone.utc)
    delta = timedelta(
        days=rng.randint(0, 30),
        hours=rng.randint(0, 23),
        minutes=rng.randint(0, 59),
    )
    tz_offsets = ["+02:00", "+05:30", "-08:00", "+00:00", "-05:00", "+09:00"]
    tz = rng.choice(tz_offsets)
    t = base - delta
    return t.strftime(f"%Y-%m-%dT%H:%M:%S{tz}")


# ---------------------------------------------------------------------------
# Trajectory templates
# ---------------------------------------------------------------------------

def tpl_set_alarm(rng: random.Random) -> dict:
    """Single tool call: set_alarm. Teaches the 'prefer specific tool' rule."""
    os_profile = rng.choice(OS_PROFILES)
    user_name = rng.choice(USER_NAMES)
    hour = rng.randint(5, 9)
    minute = rng.choice([0, 15, 30, 45])
    label = rng.choice(ALARM_LABELS)
    now_iso = random_iso_time(rng)

    tools = load_tools()
    msgs = [
        user(f"Hey Thursday, set an alarm for {hour:02d}:{minute:02d} tomorrow."),
        assistant("On it.", [call("call_001", "set_alarm", {"hour": hour, "minute": minute, "label": label})]),
        obs("set_alarm", {"ok": True, "alarm_id": f"a{rng.randint(1000,9999):x}", "scheduled_for": "2025-06-13T{:02d}:{:02d}:00+02:00".format(hour, minute), "app": "gnome-clocks" if "Linux" in os_profile else ("Clock.app" if "macOS" in os_profile else "Windows Alarms")}),
        assistant("Done.", [call("call_002", "finish", {"summary": f"Alarm set for {hour:02d}:{minute:02d} tomorrow."})]),
        obs("finish", {"ok": True}),
    ]
    return {
        "system": system_prompt(os_profile, now_iso, user_name),
        "tools": tools,
        "messages": msgs,
        "_template": "set_alarm",
    }


def tpl_open_app(rng: random.Random) -> dict:
    """Single tool call: open_app."""
    os_profile = rng.choice(OS_PROFILES)
    user_name = rng.choice(USER_NAMES)
    app = rng.choice(APPS_BY_OS[os_profile])
    now_iso = random_iso_time(rng)

    pretty_apps = {"gnome-clocks": "Clocks", "gnome-text-editor": "Text Editor", "vscode": "VS Code", "files": "Files", "explorer": "File Explorer", "clocks": "Clocks", "terminal": "Terminal", "konsole": "Konsole", "dolphin": "Dolphin", "kate": "Kate", "kmail": "KMail", "notes": "Notes", "finder": "Finder", "mail": "Mail", "notepad": "Notepad", "cmd": "Command Prompt", "powershell": "PowerShell", "safari": "Safari", "firefox": "Firefox", "chrome": "Chrome", "chromium": "Chromium", "thunderbird": "Thunderbird", "blender": "Blender", "gimp": "GIMP", "spotify": "Spotify"}
    pretty = pretty_apps.get(app, app)

    tools = load_tools()
    msgs = [
        user(f"Open {pretty} for me."),
        assistant("Opening.", [call("call_001", "open_app", {"app_name": app})]),
        obs("open_app", {"ok": True, "pid": rng.randint(1000, 99999), "window_title": pretty}),
        assistant("Done.", [call("call_002", "finish", {"summary": f"{pretty} is open."})]),
        obs("finish", {"ok": True}),
    ]
    return {
        "system": system_prompt(os_profile, now_iso, user_name),
        "tools": tools,
        "messages": msgs,
        "_template": "open_app",
    }


def tpl_set_volume(rng: random.Random) -> dict:
    """Single tool call: set_volume."""
    os_profile = rng.choice(OS_PROFILES)
    user_name = rng.choice(USER_NAMES)
    pct, desc = rng.choice(VOLUME_LEVELS)
    now_iso = random_iso_time(rng)

    tools = load_tools()
    msgs = [
        user(f"Turn the volume to {pct}."),
        assistant("Setting volume.", [call("call_001", "set_volume", {"percent": pct})]),
        obs("set_volume", {"ok": True, "percent": pct}),
        assistant("Done.", [call("call_002", "finish", {"summary": f"Volume set to {pct}%."})]),
        obs("finish", {"ok": True}),
    ]
    return {
        "system": system_prompt(os_profile, now_iso, user_name),
        "tools": tools,
        "messages": msgs,
        "_template": "set_volume",
    }


def tpl_browser_search_simple(rng: random.Random) -> dict:
    """Multi-turn: browser_search → read_webai_response → finish.
    Teaches the 'use headless when user just wants info' pattern."""
    os_profile = rng.choice(OS_PROFILES)
    user_name = rng.choice(USER_NAMES)
    engine = rng.choice(SEARCH_ENGINES)
    query = rng.choice([
        "what is the capital of Mongolia",
        "next lunar eclipse date",
        "weather in Tokyo tomorrow",
        "who wrote the song Hotel California",
        "fastest marathon time ever",
        "tallest mountain in Africa",
        "distance from earth to mars",
        "inventor of the printing press",
    ])
    now_iso = random_iso_time(rng)

    tools = load_tools()
    fake_url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
    msgs = [
        user(f"{query.capitalize()}?"),
        assistant("Searching.", [call("call_001", "browser_search", {"query": query, "engine": engine})]),
        obs("browser_search", {"ok": True, "engine": engine, "results": [{"title": f"{query} - Wikipedia", "url": fake_url, "snippet": f"...{query} is..."}], "url": f"https://{engine}.com/search?q={query.replace(' ', '+')}"}),
        assistant("Reading.", [call("call_002", "read_webai_response", {"url": fake_url})]),
        obs("read_webai_response", {"title": f"{query} - Wikipedia", "text": f"The answer to '{query}' is provided here in detail, covering history, key facts, and references.", "url": fake_url}),
        assistant("Got it.", [call("call_003", "finish", {"summary": f"Here's what I found: {query} — see the Wikipedia article I opened for the full answer."})]),
        obs("finish", {"ok": True}),
    ]
    return {
        "system": system_prompt(os_profile, now_iso, user_name),
        "tools": tools,
        "messages": msgs,
        "_template": "browser_search_simple",
    }


def tpl_open_url(rng: random.Random) -> dict:
    """Single tool: open_url. Teaches the 'direct launch' shortcut."""
    os_profile = rng.choice(OS_PROFILES)
    user_name = rng.choice(USER_NAMES)
    urls = [
        ("https://github.com/skyro777/Thursday_AI", "the Thursday AI repo"),
        ("https://youtube.com", "YouTube"),
        ("https://news.ycombinator.com", "Hacker News"),
        ("https://reddit.com/r/LocalLLaMA", "r/LocalLLaMA"),
        ("https://kaggle.com", "Kaggle"),
        ("https://huggingface.co", "Hugging Face"),
        ("https://github.com", "GitHub"),
    ]
    url, label = rng.choice(urls)
    now_iso = random_iso_time(rng)

    tools = load_tools()
    msgs = [
        user(f"Open {label}."),
        assistant("Opening.", [call("call_001", "open_url", {"url": url})]),
        obs("open_url", {"ok": True, "url": url, "launched": "default-browser"}),
        assistant("Done.", [call("call_002", "finish", {"summary": f"Opened {label} in your browser."})]),
        obs("finish", {"ok": True}),
    ]
    return {
        "system": system_prompt(os_profile, now_iso, user_name),
        "tools": tools,
        "messages": msgs,
        "_template": "open_url",
    }


def tpl_delegate_to_webai(rng: random.Random) -> dict:
    """Multi-turn: paste_to_webai → finish. Teaches delegation to a heavier AI."""
    os_profile = rng.choice(OS_PROFILES)
    user_name = rng.choice(USER_NAMES)
    now_iso = random_iso_time(rng)

    prompts = [
        ("Explain how transformers work, in one paragraph.", "Transformers are neural network architectures that use self-attention to weigh the relevance of every token in a sequence to every other token, allowing the model to capture long-range dependencies without recurrence. They consist of an encoder-decoder stack (or just one of the two, as in BERT and GPT), where each layer applies multi-head attention followed by a position-wise feedforward network, with residual connections and layer normalization. This design enables parallel training and has become the foundation of modern large language models."),
        ("Write a haiku about rain.", "Soft rain on the roof / washes the long day away / sleep comes quietly"),
        ("Name three causes of the French Revolution.", "1. Widespread inequality between the three estates, 2. Financial crisis from war debt and poor harvests, 3. Enlightenment ideas challenging the divine right of kings."),
        ("Convert 100 Celsius to Fahrenheit.", "100°C is 212°F."),
        ("Suggest a name for a friendly AI assistant.", "How about 'Thursday' — short, friendly, and easy to say."),
    ]
    user_prompt, webai_answer = rng.choice(prompts)

    tools = load_tools()
    msgs = [
        user(user_prompt),
        assistant("Thinking it through.", [call("call_001", "paste_to_webai", {"query": user_prompt})]),
        obs("paste_to_webai", {"answer": webai_answer, "model": "glm-4.5", "tokens_used": rng.randint(50, 400)}),
        assistant("Got it.", [call("call_002", "finish", {"summary": webai_answer[:200] + ("..." if len(webai_answer) > 200 else "")})]),
        obs("finish", {"ok": True}),
    ]
    return {
        "system": system_prompt(os_profile, now_iso, user_name),
        "tools": tools,
        "messages": msgs,
        "_template": "delegate_to_webai",
    }


def tpl_world_news(rng: random.Random) -> dict:
    """Multi-turn: paste_to_webai → ask_user → read_webai_response → finish.
    Teaches the 'show headlines, ask which to read' pattern."""
    os_profile = rng.choice(OS_PROFILES)
    user_name = rng.choice(USER_NAMES)
    now_iso = random_iso_time(rng)
    query = rng.choice(NEWS_QUERIES)

    tools = load_tools()
    headlines = [
        ("Climate summit reaches surprise agreement", "https://example-news.com/climate", "Negotiators finalized a framework to phase out coal by 2035..."),
        ("Major tech outage hits airlines", "https://example-news.com/outage", "A software update grounded flights worldwide..."),
        ("Mars sample return delayed to 2030", "https://example-news.com/mars", "NASA announced a two-year slip in the mission..."),
    ]
    pick = rng.choice(headlines)
    full_text = f"Top story: {pick[0]}. {pick[2]} Read more at {pick[1]}"

    msgs = [
        user("What's happening in the world right now?"),
        assistant("Checking news.", [call("call_001", "paste_to_webai", {"query": query, "num": 8})]),
        obs("paste_to_webai", {"results": [{"title": h[0], "url": h[1], "snippet": h[2]} for h in headlines], "engine": "duckduckgo"}),
        assistant("Want details?", [call("call_002", "ask_user", {"question": "Want me to read any of these in detail?"})]),
        obs("ask_user", {"response": "yes the first one"}),
        assistant("Reading it.", [call("call_003", "read_webai_response", {"url": pick[1]})]),
        obs("read_webai_response", {"title": pick[0], "text": pick[2], "url": pick[1]}),
        assistant("Done.", [call("call_004", "finish", {"summary": full_text})]),
        obs("finish", {"ok": True}),
    ]
    return {
        "system": system_prompt(os_profile, now_iso, user_name),
        "tools": tools,
        "messages": msgs,
        "_template": "world_news",
    }


def tpl_explain_file(rng: random.Random) -> dict:
    """Multi-turn: read_file → finish. Teaches the 'read and summarize' pattern."""
    os_profile = rng.choice(OS_PROFILES)
    user_name = rng.choice(USER_NAMES)
    now_iso = random_iso_time(rng)
    path = rng.choice(FILE_PATHS)

    files = {
        "~/Downloads/notes.txt": ("Meeting notes:\n- ship v0.1 by Friday\n- ask Riya about the dataset license\n- buy milk", "Meeting notes — three items: ship v0.1, ask Riya about a dataset license, buy milk."),
        "~/Documents/todo.md": ("# TODO\n- [x] clone repo\n- [ ] write plan\n- [ ] write generator\n- [ ] run on Kaggle", "A 4-item TODO list for the Thursday AI project, two items still pending."),
        "~/projects/main.py": ("import sys\nprint('hello world')\n", "A 2-line hello-world Python script."),
        "~/Pictures/vacation.txt": ("Trip to Goa\nDay 1: beach\nDay 2: fort\nDay 3: spice farm", "Notes from a 3-day Goa trip: beach, fort, spice farm."),
        "~/Desktop/scratch.txt": ("test 1 2 3\n", "Just a 1-line test scratch file."),
        "~/Music/playlist.txt": ("1. Blinding Lights\n2. Levitating\n3. Perfect\n", "A 3-song playlist."),
    }
    content, summary = files[path]

    tools = load_tools()
    msgs = [
        user(f"What's in {path}?"),
        assistant("Reading it.", [call("call_001", "read_file", {"path": path})]),
        obs("read_file", {"ok": True, "path": path, "content": content, "bytes": len(content)}),
        assistant("Got it.", [call("call_002", "finish", {"summary": summary})]),
        obs("finish", {"ok": True}),
    ]
    return {
        "system": system_prompt(os_profile, now_iso, user_name),
        "tools": tools,
        "messages": msgs,
        "_template": "explain_file",
    }


def tpl_ask_user_clarify(rng: random.Random) -> dict:
    """Single tool: ask_user, then finish. Teaches clarification pattern."""
    os_profile = rng.choice(OS_PROFILES)
    user_name = rng.choice(USER_NAMES)
    now_iso = random_iso_time(rng)

    ambiguous = [
        ("Play some music.", "Which artist or genre did you have in mind?"),
        ("Set an alarm.", "Sure — what time should I set it for?"),
        ("Open the file.", "Which file did you mean?"),
        ("Search for it.", "What should I search for?"),
        ("Send the message.", "Who should I send it to, and what should I say?"),
        ("Make a reminder.", "What should the reminder be, and when?"),
    ]
    user_msg, question = rng.choice(ambiguous)

    tools = load_tools()
    msgs = [
        user(user_msg),
        assistant("Just checking.", [call("call_001", "ask_user", {"question": question})]),
        obs("ask_user", {"response": "never mind"}),
        assistant("Done.", [call("call_002", "finish", {"summary": "Cancelled — no action taken."})]),
        obs("finish", {"ok": True}),
    ]
    return {
        "system": system_prompt(os_profile, now_iso, user_name),
        "tools": tools,
        "messages": msgs,
        "_template": "ask_user_clarify",
    }


def tpl_list_dir(rng: random.Random) -> dict:
    """Multi-turn: list_dir → finish."""
    os_profile = rng.choice(OS_PROFILES)
    user_name = rng.choice(USER_NAMES)
    now_iso = random_iso_time(rng)
    path = rng.choice(["~/Downloads", "~/Documents", "~/projects", "~/Pictures", "~", "~/Desktop"])

    listings = {
        "~/Downloads": [("cleanup.py", "file", 247), ("report.pdf", "file", 102400), ("music", "dir", 0), ("vacation.txt", "file", 80)],
        "~/Documents": [("todo.md", "file", 120), ("resume.pdf", "file", 51200), ("work", "dir", 0)],
        "~/projects": [("thursday-ai", "dir", 0), ("notes.txt", "file", 350)],
        "~/Pictures": [("vacation", "dir", 0), ("avatar.png", "file", 4096)],
        "~": [(".thursday", "dir", 0), ("Downloads", "dir", 0), ("Documents", "dir", 0), ("projects", "dir", 0), (".bashrc", "file", 1024)],
        "~/Desktop": [("scratch.txt", "file", 20), ("notes.md", "file", 80)],
    }
    items = listings[path]

    tools = load_tools()
    msgs = [
        user(f"What's in {path}?"),
        assistant("Listing.", [call("call_001", "list_dir", {"path": path})]),
        obs("list_dir", {"ok": True, "path": path, "entries": [{"name": n, "type": t, "size_bytes": s} for n, t, s in items]}),
        assistant("Done.", [call("call_002", "finish", {"summary": f"{path} has {len(items)} items: " + ", ".join(n for n, _, _ in items) + "."})]),
        obs("finish", {"ok": True}),
    ]
    return {
        "system": system_prompt(os_profile, now_iso, user_name),
        "tools": tools,
        "messages": msgs,
        "_template": "list_dir",
    }


# Registry
TEMPLATES: dict[str, Callable[[random.Random], dict]] = {
    "set_alarm": tpl_set_alarm,
    "open_app": tpl_open_app,
    "set_volume": tpl_set_volume,
    "browser_search_simple": tpl_browser_search_simple,
    "open_url": tpl_open_url,
    "delegate_to_webai": tpl_delegate_to_webai,
    "world_news": tpl_world_news,
    "explain_file": tpl_explain_file,
    "ask_user_clarify": tpl_ask_user_clarify,
    "list_dir": tpl_list_dir,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(count: int, seed: int, output_path: Path, weights: dict[str, float] | None = None) -> int:
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Default weights: roughly even across templates, with extra weight on the
    # multi-turn ones since those teach the most.
    if weights is None:
        weights = {
            "set_alarm": 1.0,
            "open_app": 1.0,
            "set_volume": 0.7,
            "browser_search_simple": 1.2,
            "open_url": 1.0,
            "delegate_to_webai": 1.3,
            "world_news": 1.0,
            "explain_file": 1.0,
            "ask_user_clarify": 0.8,
            "list_dir": 0.8,
        }
    total_w = sum(weights.values())
    template_names = list(weights.keys())
    template_weights = [weights[n] / total_w for n in template_names]

    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for _ in range(count):
            tpl_name = rng.choices(template_names, weights=template_weights, k=1)[0]
            example = TEMPLATES[tpl_name](rng)
            # Strip the _template marker before writing (it's metadata for us)
            example_for_disk = {k: v for k, v in example.items() if k != "_template"}
            f.write(json.dumps(example_for_disk, ensure_ascii=False) + "\n")
            written += 1
            if written % 5000 == 0:
                print(f"  ... wrote {written}/{count}", file=sys.stderr)

    print(f"Wrote {written} examples to {output_path}", file=sys.stderr)
    return written


def main():
    p = argparse.ArgumentParser(description="Thursday AI Method A dataset generator")
    p.add_argument("--count", type=int, default=1000, help="Number of examples to generate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=str(OUTPUT_DEFAULT))
    p.add_argument("--list-templates", action="store_true")
    args = p.parse_args()

    if args.list_templates:
        print("Available templates:")
        for name, fn in TEMPLATES.items():
            print(f"  {name}")
        return

    generate(count=args.count, seed=args.seed, output_path=Path(args.out))


if __name__ == "__main__":
    main()
