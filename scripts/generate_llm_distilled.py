#!/usr/bin/env python3
"""
generate_llm_distilled.py — Thursday AI Method B dataset generator.

Uses a stronger "teacher" model to generate diverse agentic trajectories from
seed examples, via Evol-Instruct-style prompt mutation.

Output: data/processed/llm_distilled.jsonl

Teacher model options:
  - Local Qwen2.5-7B-Instruct via Ollama (default; free, offline)
  - GLM-4.5 / GLM-4.6 via z-ai-web-dev-sdk (free tier; needs ZAI_API_KEY)
  - Any OpenAI-compatible endpoint via --api-base + --api-key

Usage:
    # Default: use local Ollama
    python scripts/generate_llm_distilled.py --count 200 --seeds data/examples/*.jsonl

    # Use ZAI SDK
    python scripts/generate_llm_distilled.py --count 200 --teacher zai --seeds data/examples/*.jsonl

    # Use custom OpenAI-compatible endpoint
    python scripts/generate_llm_distilled.py --count 200 \\
        --teacher openai \\
        --api-base https://api.example.com/v1 \\
        --api-key sk-xxx \\
        --model gpt-4o-mini \\
        --seeds data/examples/*.jsonl

This script DOES NOT write code itself — it prompts the teacher to produce
JSONL-formatted examples, validates each one with the same logic as
validate_dataset.py, and rejects invalid outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "data" / "schema" / "tools.json"
OUTPUT_DEFAULT = REPO_ROOT / "data" / "processed" / "llm_distilled.jsonl"

# Import the validator + template generator helpers
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from validate_dataset import validate_example  # noqa: E402
from generate_template_data import load_tools, system_prompt, OS_PROFILES, USER_NAMES, random_iso_time  # noqa: E402

# ---------------------------------------------------------------------------
# Evol-Instruct style mutation prompts
# ---------------------------------------------------------------------------

MUTATION_STYLES = [
    "Make the user's request more specific: add a constraint (time, location, quantity, or preference).",
    "Make the user's request more complex: chain two related goals into one utterance.",
    "Rephrase the user's request in casual spoken English (contractions, slang, hedging).",
    "Make the user's request imply a different OS context (Linux vs Windows vs macOS).",
    "Add a slight ambiguity that requires the assistant to call ask_user once.",
    "Rephrase the request as if the user is mid-task and continuing from a previous step.",
    "Add a constraint that requires the assistant to use paste_to_webai to look something up first.",
    "Make the user explicitly want to watch the browser being driven (not headless).",
]

EVOL_PROMPT_TEMPLATE = """You are a data-generation teacher for "Thursday AI", an offline voice-based AI that controls the user's PC through tool calls.

You will receive ONE seed trajectory in JSONL format and a mutation instruction. Your job:
1. Apply the mutation instruction to the user's request and re-plan a realistic, valid trajectory that uses the SAME tool schema.
2. Output ONE new example in the EXACT same JSON schema: {{"system": str, "tools": list, "messages": list}}.
3. Each assistant turn MUST contain at most ONE tool_calls entry.
4. Every tool_call.function.name MUST exist in the tools array.
5. tool_call.arguments MUST be a JSON string (not a JSON object).
6. Every tool_call id MUST be followed by a matching tool-role message whose `name` matches the tool name.
7. Use realistic OCR text in screenshot observations.
8. Keep assistant spoken replies short (3-7 words) while working; final finish() summary ≤ 2 lines.
9. Use the same system prompt structure as the seed but vary OS, time, and user_name.
10. Do NOT include code fences (```) in your output. Output only the raw JSON object on one line.

SEED (do not return the seed; produce a NEW variant):
{seed}

MUTATION INSTRUCTION: {mutation}

NOW produce the new example as a single JSON line. Begin with {{ and end with }}.
"""

# ---------------------------------------------------------------------------
# Teacher adapters
# ---------------------------------------------------------------------------

class Teacher:
    """Base class — subclasses implement `generate(prompt) -> str`."""

    def generate(self, prompt: str, max_tokens: int = 1500) -> str:
        raise NotImplementedError


class OllamaTeacher(Teacher):
    """Use a local Ollama instance."""

    def __init__(self, model: str = "qwen2.5:7b-instruct-q4_K_M"):
        self.model = model
        self.api_base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def generate(self, prompt: str, max_tokens: int = 1500) -> str:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.8, "top_p": 0.9, "num_predict": max_tokens},
        }).encode()
        req = urllib.request.Request(
            f"{self.api_base}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
                return data.get("response", "")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama not reachable at {self.api_base}: {e}. Did you `ollama serve`?")


class ZAITeacher(Teacher):
    """Use the z-ai-web-dev-sdk via its CLI / SDK. The SDK exposes a chat.completions
    endpoint. Here we shell out to the `z-ai` CLI if available, falling back to
    the SDK if imported in-process."""

    def __init__(self, model: str = "glm-4.6"):
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 1500) -> str:
        # Try the in-process SDK first
        try:
            from zai import ZaiClient  # type: ignore
            client = ZaiClient()  # reads ZAI_API_KEY from env
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.8,
            )
            return resp.choices[0].message.content
        except ImportError:
            pass
        # Fallback: shell out to the CLI
        try:
            result = subprocess.run(
                ["z-ai", "chat", "--model", self.model, "--prompt", prompt, "--max-tokens", str(max_tokens)],
                capture_output=True, text=True, timeout=180,
            )
            if result.returncode != 0:
                raise RuntimeError(f"z-ai CLI failed: {result.stderr}")
            return result.stdout
        except FileNotFoundError:
            raise RuntimeError(
                "Neither zai SDK nor z-ai CLI is available. "
                "Set ZAI_API_KEY and `pip install z-ai-web-dev-sdk`, or use --teacher ollama."
            )


class OpenAITeacher(Teacher):
    """Generic OpenAI-compatible endpoint."""

    def __init__(self, model: str, api_base: str, api_key: str):
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key

    def generate(self, prompt: str, max_tokens: int = 1500) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.8,
        }).encode()
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# JSON extraction (strip code fences + find the outermost {})
# ---------------------------------------------------------------------------

def extract_json_object(text: str) -> str | None:
    """Find the first balanced {} in text. Returns the substring or None."""
    text = text.strip()
    # Strip ```json ... ``` fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Find first { and balance
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def load_seeds(seed_paths: list[str]) -> list[dict]:
    seeds = []
    for sp in seed_paths:
        p = Path(sp)
        if p.is_dir():
            for jf in sorted(p.glob("*.jsonl")):
                with jf.open() as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            seeds.append(json.loads(line))
        else:
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        seeds.append(json.loads(line))
    return seeds


def main():
    p = argparse.ArgumentParser(description="Thursday AI Method B — LLM-distilled dataset generator")
    p.add_argument("--count", type=int, default=200, help="Number of new examples to generate")
    p.add_argument("--seeds", nargs="+", required=True, help="JSONL files/dirs of seed examples")
    p.add_argument("--out", type=str, default=str(OUTPUT_DEFAULT))
    p.add_argument("--teacher", choices=["ollama", "zai", "openai"], default="ollama")
    p.add_argument("--model", type=str, default=None, help="Override teacher model name")
    p.add_argument("--api-base", type=str, default=None)
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-retries", type=int, default=2, help="Retries per example if teacher output fails validation")
    args = p.parse_args()

    rng = random.Random(args.seed)

    # Build teacher
    if args.teacher == "ollama":
        teacher = OllamaTeacher(model=args.model or "qwen2.5:7b-instruct-q4_K_M")
    elif args.teacher == "zai":
        teacher = ZAITeacher(model=args.model or "glm-4.6")
    else:
        if not (args.api_base and args.api_key):
            print("ERROR: --teacher openai requires --api-base and --api-key", file=sys.stderr)
            sys.exit(2)
        teacher = OpenAITeacher(model=args.model or "gpt-4o-mini", api_base=args.api_base, api_key=args.api_key)

    seeds = load_seeds(args.seeds)
    if not seeds:
        print("ERROR: no seed examples found", file=sys.stderr)
        sys.exit(2)
    print(f"Loaded {len(seeds)} seed examples", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    rejected = 0
    with out_path.open("w", encoding="utf-8") as f:
        attempts = 0
        while written < args.count and attempts < args.count * 3:
            attempts += 1
            seed = rng.choice(seeds)
            mutation = rng.choice(MUTATION_STYLES)
            prompt = EVOL_PROMPT_TEMPLATE.format(
                seed=json.dumps(seed, ensure_ascii=False)[:4000],
                mutation=mutation,
            )

            try:
                raw = teacher.generate(prompt, max_tokens=2000)
            except Exception as e:
                print(f"  [retry] teacher error: {e}", file=sys.stderr)
                time.sleep(1)
                continue

            json_str = extract_json_object(raw)
            if json_str is None:
                rejected += 1
                continue

            ok, errors = validate_example(line_no=0, raw=json_str)
            if not ok:
                rejected += 1
                # Try once more with a reminder appended
                if args.max_retries > 0:
                    retry_prompt = prompt + "\n\nYour previous output was INVALID for these reasons:\n- " + "\n- ".join(errors[:3]) + "\n\nPlease output a SINGLE valid JSON line now."
                    try:
                        raw2 = teacher.generate(retry_prompt, max_tokens=2000)
                        json_str2 = extract_json_object(raw2)
                        if json_str2:
                            ok2, errors2 = validate_example(line_no=0, raw=json_str2)
                            if ok2:
                                f.write(json_str2 + "\n")
                                written += 1
                                continue
                    except Exception:
                        pass
                continue

            f.write(json_str + "\n")
            written += 1
            if written % 25 == 0:
                print(f"  ... wrote {written}/{args.count} (rejected {rejected})", file=sys.stderr)

    print(f"Done. Wrote {written} valid examples to {out_path} (rejected {rejected}).", file=sys.stderr)


if __name__ == "__main__":
    main()
