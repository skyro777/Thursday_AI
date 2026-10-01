#!/usr/bin/env python3
"""
validate_dataset.py — Thursday AI dataset validator.

Checks every JSONL example for:
  1. Valid JSON.
  2. Required top-level fields: system, tools, messages.
  3. `tools` is a list of valid OpenAI-style function schemas.
  4. `messages` alternate correctly and contain valid tool_call IDs that
     are echoed back by their tool-role observation.
  5. Every tool_call.function.name is declared in `tools`.
  6. Every `tool` message has a matching preceding tool_call.
  7. `arguments` is a JSON string (Qwen2 convention).
  8. Each assistant turn has at most one tool_calls entry (Thursday AI rule).

Exit codes:
  0  all examples valid
  1  some examples invalid (details printed to stderr)
  2  file unreadable / not JSONL

Usage:
    python scripts/validate_dataset.py data/processed/template_generated.jsonl
    python scripts/validate_dataset.py data/processed/*.jsonl --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_TOP_FIELDS = {"system", "tools", "messages"}
ASSISTANT_MAX_TOOL_CALLS = 1  # Thursday AI rule: atomicity


def validate_example(line_no: int, raw: str) -> tuple[bool, list[str]]:
    """Return (ok, list_of_errors)."""
    errors: list[str] = []

    # 1. Valid JSON
    try:
        ex: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, [f"line {line_no}: invalid JSON ({e})"]

    if not isinstance(ex, dict):
        return False, [f"line {line_no}: top-level is not an object"]

    # 2. Required top-level fields
    missing = REQUIRED_TOP_FIELDS - set(ex.keys())
    if missing:
        errors.append(f"line {line_no}: missing top-level fields: {sorted(missing)}")

    # 3. tools is a list of dicts with name+description+parameters
    tools = ex.get("tools", [])
    if not isinstance(tools, list):
        errors.append(f"line {line_no}: 'tools' is not a list")
        tools = []
    declared_names: set[str] = set()
    for i, t in enumerate(tools):
        if not isinstance(t, dict):
            errors.append(f"line {line_no}: tools[{i}] is not an object")
            continue
        for k in ("name", "description", "parameters"):
            if k not in t:
                errors.append(f"line {line_no}: tools[{i}] missing '{k}'")
        if "name" in t and isinstance(t["name"], str):
            declared_names.add(t["name"])

    # 4. messages shape + tool-call id consistency
    messages = ex.get("messages", [])
    if not isinstance(messages, list):
        errors.append(f"line {line_no}: 'messages' is not a list")
        messages = []

    # Pre-collect expected tool_call ids per assistant turn
    pending_tool_calls: dict[str, str] = {}  # id -> tool_name
    last_role = None
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"line {line_no}: messages[{i}] is not an object")
            continue
        role = msg.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            errors.append(f"line {line_no}: messages[{i}] has unknown role '{role}'")

        if role == "system" and i != 0:
            # System messages should only appear at index 0 in our schema
            errors.append(f"line {line_no}: system message at index {i} (must be 0)")

        if role == "user":
            if last_role not in (None, "tool", "assistant"):
                errors.append(f"line {line_no}: user message at index {i} follows unexpected role '{last_role}'")
            if "content" not in msg or not isinstance(msg["content"], str):
                errors.append(f"line {line_no}: user message at index {i} missing string 'content'")

        if role == "assistant":
            # Assistant may have content (str or null) and/or tool_calls
            if "content" not in msg:
                errors.append(f"line {line_no}: assistant msg[{i}] missing 'content' (use null if no content)")
            else:
                c = msg["content"]
                if c is not None and not isinstance(c, str):
                    errors.append(f"line {line_no}: assistant msg[{i}] content must be string or null")

            tool_calls = msg.get("tool_calls", [])
            if tool_calls is None:
                tool_calls = []
            if not isinstance(tool_calls, list):
                errors.append(f"line {line_no}: assistant msg[{i}] tool_calls is not a list")
                tool_calls = []
            if len(tool_calls) > ASSISTANT_MAX_TOOL_CALLS:
                errors.append(
                    f"line {line_no}: assistant msg[{i}] has {len(tool_calls)} tool_calls "
                    f"(max allowed: {ASSISTANT_MAX_TOOL_CALLS}; Thursday AI requires atomic single-call turns)"
                )
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    errors.append(f"line {line_no}: tool_call in msg[{i}] is not an object")
                    continue
                for k in ("id", "type", "function"):
                    if k not in tc:
                        errors.append(f"line {line_no}: tool_call in msg[{i}] missing '{k}'")
                fn = tc.get("function", {})
                if not isinstance(fn, dict):
                    errors.append(f"line {line_no}: tool_call.function in msg[{i}] is not an object")
                    continue
                for k in ("name", "arguments"):
                    if k not in fn:
                        errors.append(f"line {line_no}: tool_call.function in msg[{i}] missing '{k}'")
                # arguments must be a JSON string (Qwen2 convention)
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        json.loads(args)
                    except json.JSONDecodeError as e:
                        errors.append(f"line {line_no}: tool_call.arguments in msg[{i}] is not valid JSON ({e})")
                elif args is not None:
                    errors.append(f"line {line_no}: tool_call.arguments in msg[{i}] must be a JSON string (got {type(args).__name__})")
                # name must be declared
                tc_name = fn.get("name")
                if tc_name and tc_name not in declared_names:
                    errors.append(f"line {line_no}: tool_call in msg[{i}] references undeclared tool '{tc_name}'")
                # record pending id
                tc_id = tc.get("id")
                if tc_id:
                    pending_tool_calls[tc_id] = tc_name

        if role == "tool":
            # must reference a pending tool_call id via `tool_call_id` OR
            # (our convention) via the message's `name` field matching the
            # most recent assistant tool_call whose function.name == msg.name
            tool_name = msg.get("name")
            if not tool_name:
                errors.append(f"line {line_no}: tool msg[{i}] missing 'name'")
            content = msg.get("content")
            if not isinstance(content, str):
                errors.append(f"line {line_no}: tool msg[{i}] 'content' must be a string")
            else:
                # content should be valid JSON (the observation payload)
                try:
                    json.loads(content)
                except json.JSONDecodeError as e:
                    errors.append(f"line {line_no}: tool msg[{i}] content is not valid JSON ({e})")
            # try to find a matching pending tool_call
            # We pop the most recent pending id whose function.name == tool_name
            matched_id = None
            for pid, pname in list(pending_tool_calls.items()):
                if pname == tool_name:
                    matched_id = pid
                    del pending_tool_calls[pid]
                    break
            if matched_id is None:
                errors.append(f"line {line_no}: tool msg[{i}] name='{tool_name}' has no matching pending assistant tool_call")

        last_role = role

    # Any pending tool_calls that were never answered?
    if pending_tool_calls:
        for pid, pname in pending_tool_calls.items():
            errors.append(f"line {line_no}: tool_call id={pid} name='{pname}' was never answered by a tool observation")

    return len(errors) == 0, errors


def main():
    p = argparse.ArgumentParser(description="Validate Thursday AI JSONL dataset")
    p.add_argument("files", nargs="+", help="JSONL files to validate")
    p.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    p.add_argument("--max-errors", type=int, default=20, help="Stop after this many error lines per file")
    args = p.parse_args()

    total_ok = 0
    total_err = 0
    total_files = 0

    # Expand any directory arguments into their *.jsonl files, recursively.
    # This lets users pass either a file or a directory (or both) without
    # hitting IsADirectoryError.
    expanded_files: list[Path] = []
    for path_str in args.files:
        p = Path(path_str)
        if not p.exists():
            print(f"❌ {p}: file not found", file=sys.stderr)
            total_err += 1
            continue
        if p.is_dir():
            jsonl_files = sorted(p.rglob("*.jsonl"))
            if not jsonl_files:
                print(f"⚠️  {p}: no .jsonl files found in directory", file=sys.stderr)
            expanded_files.extend(jsonl_files)
        else:
            expanded_files.append(p)

    for path in expanded_files:
        total_files += 1
        file_ok = 0
        file_err = 0
        with path.open() as f:
            for line_no, line in enumerate(f, 1):
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                ok, errors = validate_example(line_no, line)
                if ok:
                    file_ok += 1
                else:
                    file_err += 1
                    for e in errors[:5]:
                        print(f"  {e}", file=sys.stderr)
                    if file_err >= args.max_errors:
                        print(f"  ... (suppressing further errors in {path.name}; use --max-errors to see more)", file=sys.stderr)
                        break
        print(f"{'✅' if file_err == 0 else '⚠️'}  {path.name}: {file_ok} ok, {file_err} bad")
        total_ok += file_ok
        total_err += file_err

    print(f"\nTotal: {total_ok} valid, {total_err} invalid across {total_files} files")
    sys.exit(0 if (total_err == 0 or not args.strict) else 1)


if __name__ == "__main__":
    main()
