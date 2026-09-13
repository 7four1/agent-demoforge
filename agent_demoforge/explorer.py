"""The Explore phase: an agentic tool-use loop that reads the target repo.

This is the differentiator from a generic README-summarizer: the model is
instructed to actually inspect source files (entry points, core modules)
rather than just repeating the README's prose, using two path-confined
tools: `list_dir` and `read_file`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .sandbox import PathEscapeError, safe_join

MAX_READ_CHARS = 8000
MAX_ITERATIONS = 15

SYSTEM_PROMPT = """You are a senior engineer preparing a short narrated video demo of a code \
repository, by actually reading its code -- not by repeating marketing copy.

You have two tools, both confined to the target repository's root directory:
- list_dir(path): list files/subdirectories directly inside a directory ('.' is the root).
- read_file(path, start_line=None, end_line=None): read a text file, optionally a line range.

Explore in this order:
1. Read the README (or equivalent) first, to learn what the project claims to be.
2. Look at packaging/entry-point files (pyproject.toml, setup.py, package.json, Makefile,
   __main__.py, cli.py, main.py, index.js, etc.) to find how the project is actually run.
3. Read the real, core source files that implement the project's main functionality --
   enough of them to genuinely understand what the code does and how data flows through it.

Your job is to form your own source-grounded understanding of what this project really does,
including anything the README undersells, omits, or gets a little too marketing-y about.
Do not just paraphrase the README back -- verify its claims against the actual code, and
identify concrete, runnable commands (installs, CLI invocations, test runs, etc.) that would
make a good, genuine demonstration of the project's real behavior.

Keep exploring only as long as it adds real understanding -- a handful of well-chosen
list_dir/read_file calls is usually enough for a small project. When you have enough of a
grounded understanding, stop calling tools and reply with a concise plain-text summary
covering: (1) what the project actually does, (2) its real entry point(s) and how a user
would run it, (3) 2-4 concrete, runnable commands worth demonstrating and why."""

EXPLORE_INSTRUCTION = (
    "Explore this repository and build a genuine understanding of what it does, "
    "grounded in the actual source code -- not just the README."
)

TOOLS = [
    {
        "name": "list_dir",
        "description": (
            "List the files and subdirectories directly inside a directory of the "
            "target repository. Path is relative to the repository root; use '.' "
            "for the root."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to the repo root, e.g. '.' or 'src'.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read the contents of a text file in the target repository, optionally "
            "restricted to a line range. Path is relative to the repository root."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the repo root.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "1-indexed first line to include (optional).",
                },
                "end_line": {
                    "type": "integer",
                    "description": "1-indexed last line to include, inclusive (optional).",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
]


def list_dir(repo_root: str, path: str = ".") -> str:
    try:
        target = safe_join(repo_root, path)
    except PathEscapeError as e:
        return f"Error: {e}"
    if not os.path.isdir(target):
        return f"Error: '{path}' is not a directory"
    entries = []
    for name in sorted(os.listdir(target)):
        if name in (".git", "__pycache__", ".agent_demoforge_venv", ".venv", "venv"):
            continue
        full = os.path.join(target, name)
        if os.path.isdir(full):
            entries.append(f"dir   {name}/")
        else:
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            entries.append(f"file  {name}  ({size} bytes)")
    return "\n".join(entries) if entries else "(empty directory)"


def read_file(
    repo_root: str,
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> str:
    try:
        target = safe_join(repo_root, path)
    except PathEscapeError as e:
        return f"Error: {e}"
    if not os.path.isfile(target):
        return f"Error: '{path}' is not a file"
    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        return f"Error reading file: {e}"

    total = len(lines)
    start = max(1, start_line or 1)
    end = min(total, end_line if end_line is not None else total)
    selected = lines[start - 1 : end] if start <= end else []
    text = "".join(selected)

    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + f"\n... [truncated at {MAX_READ_CHARS} chars]"
    return text or "(empty file or empty line range)"


def read_code_excerpt(
    repo_root: str,
    path: str,
    start_line: Optional[int],
    end_line: Optional[int],
):
    """Read a real, clamped line range of a real file for rendering a
    code_walkthrough frame. Unlike `read_file` (which returns a
    human-readable "Error: ..." string for the LLM), this returns None on
    any failure -- missing file, path escaping the sandbox, unreadable file,
    or an empty file -- so callers can cleanly skip the code frame instead
    of ever trusting LLM-provided text as the on-screen code.

    On success returns (clamped_start_line, clamped_end_line, text): a
    hallucinated out-of-bounds line range is clamped to the file's real
    bounds rather than causing a crash.
    """
    try:
        target = safe_join(repo_root, path)
    except PathEscapeError:
        return None
    if not os.path.isfile(target):
        return None
    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None

    total = len(lines)
    if total == 0:
        return None

    start = start_line if start_line and start_line > 0 else 1
    start = min(start, total)
    end = end_line if end_line and end_line > 0 else total
    end = min(end, total)
    if end < start:
        end = start

    selected = lines[start - 1 : end]
    text = "".join(selected)
    return start, end, text


def execute_tool(repo_root: str, name: str, tool_input: Dict[str, Any]) -> str:
    if name == "list_dir":
        return list_dir(repo_root, tool_input.get("path", "."))
    if name == "read_file":
        return read_file(
            repo_root,
            tool_input.get("path", "."),
            tool_input.get("start_line"),
            tool_input.get("end_line"),
        )
    return f"Error: unknown tool '{name}'"


def run_explore_loop(
    client,
    model: str,
    repo_root: str,
    max_iterations: int = MAX_ITERATIONS,
) -> List[Dict[str, Any]]:
    """Manual agentic tool-use loop. Returns the full message transcript,
    which the scriptwriter phase continues from."""
    messages: List[Dict[str, Any]] = [{"role": "user", "content": EXPLORE_INSTRUCTION}]

    for _ in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue

        if response.stop_reason == "end_turn":
            messages.append({"role": "assistant", "content": response.content})
            break

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": response.content})

        if not tool_use_blocks:
            # No tool calls and not end_turn (e.g. max_tokens, refusal) -- stop here.
            break

        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": t.id,
                "content": execute_tool(repo_root, t.name, t.input),
            }
            for t in tool_use_blocks
        ]
        messages.append({"role": "user", "content": tool_results})

    return messages
