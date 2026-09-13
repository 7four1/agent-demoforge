"""Interactive Q&A about a target repository: `agent-demoforge ask` (one-shot)
and `agent-demoforge chat` (multi-turn REPL).

Much simpler safety story than `generate`: these commands are entirely
read-only. They give the model exactly the same two path-confined tools as
the Explore phase (`explorer.TOOLS` / `explorer.execute_tool` --
`list_dir`/`read_file`), reused as-is, and never execute a shell command, so
there is no `--yes` gate to clear.

Bidirectional Explore-phase cache sharing with `generate` is the point of
this module:
  - If a cached Explore transcript already exists for this repo content +
    model (written by a prior `generate` run, or a prior `ask`/`chat`
    run), it's loaded and reused as a head start -- the question is simply
    appended as a new turn on top of it (`cache.load`, same cache as
    `pipeline.run_llm_phases`).
  - If there's no cache entry, this module runs the exact same
    `explorer.run_explore_loop` that `generate` uses, then best-effort
    saves that transcript with `cache.save` -- so a LATER `generate` run
    (or another `ask`/`chat`) against the same repo+model gets a cache hit
    too.

Unlike `generate`, there is NO offline fallback here: open-ended Q&A with
no live model connection has no honest heuristic substitute (there's no
`build_fallback_script`-style equivalent for "answer an arbitrary
question"). `check_credentials()` fails fast and clean, before any sandbox
is even prepared, when no live Anthropic connection is available.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import cache as cache_mod
from . import explorer, sandbox
from . import tts as tts_mod

MAX_QA_ITERATIONS = 10
MAX_SPEAK_CHARS = 2000

QA_SYSTEM_PROMPT = """You are a senior engineer answering questions about a code repository, \
grounded in the actual code -- never guesses, never generic assumptions, never repeating \
marketing copy from the README as if it were verified fact.

You have two tools, both confined to the target repository's root directory:
- list_dir(path): list files/subdirectories directly inside a directory ('.' is the root).
- read_file(path, start_line=None, end_line=None): read a text file, optionally a line range.

No shell commands are ever executed -- these two tools only ever read the filesystem.

If earlier turns in this conversation already contain an exploration of this repository, \
use that understanding, and only call more tools if the current question needs something \
not already covered or a closer look at something specific. If there is no prior \
exploration, use the tools to investigate before answering.

Answer the user's question directly and concisely, grounded in what you have actually read \
in the code -- cite file paths when it's useful. If the answer genuinely isn't determinable \
from the repository, say so plainly rather than guessing. When you are confident in your \
answer, stop calling tools and reply with your final answer as plain text meant to be read \
(and optionally spoken aloud) directly -- no markdown formatting."""


def check_credentials() -> Tuple[Optional[Any], Optional[str]]:
    """Return (client, reason). `client` is None and `reason` explains why
    when a live Anthropic connection isn't available. Checked up front, in
    both `run_ask` and `run_chat`, before any sandbox is prepared -- unlike
    `generate`, there is no offline fallback to degrade to here, so failing
    fast and cleanly matters.

    This mirrors the same "is the SDK importable" check `pipeline.py` makes
    before its own try/except-broadened live-call attempt, plus an
    explicit `ANTHROPIC_API_KEY` presence check so the common "forgot to
    set it" case is reported immediately and specifically rather than only
    surfacing later as an opaque authentication error from the first API
    call.
    """
    try:
        import anthropic
    except ImportError:
        return None, "the 'anthropic' package is not installed"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None, "no ANTHROPIC_API_KEY environment variable is set"
    return anthropic.Anthropic(), None


def _block_type(block: Any) -> Optional[str]:
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _block_attr(block: Any, name: str) -> Any:
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)


def extract_answer_text(content_blocks: Optional[Sequence[Any]]) -> str:
    """Concatenate the text blocks of a final assistant turn into the
    plain-text answer. Handles both live SDK content-block objects
    (attribute access) and plain dicts (e.g. if a caller passes already-
    converted content), matching the tolerant style of `cache._to_plain`."""
    if not content_blocks:
        return ""
    texts = [
        _block_attr(b, "text")
        for b in content_blocks
        if _block_type(b) == "text" and _block_attr(b, "text")
    ]
    return "\n".join(texts).strip()


def answer_question(
    client,
    model: str,
    repo_root: str,
    messages: List[Dict[str, Any]],
    question: str,
    max_iterations: int = MAX_QA_ITERATIONS,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Answer `question`, continuing from `messages` (an existing transcript
    -- cached Explore output, a fresh Explore transcript, or prior chat
    turns). Returns (answer_text, updated_messages); `updated_messages`
    includes the question, every tool_use/tool_result round, and the final
    answer, so a caller (`chat`) can pass it straight back in as the next
    turn's `messages` to preserve full conversation history.

    Same manual agentic tool-use loop shape as `explorer.run_explore_loop`,
    capped at `max_iterations` (default 10, deliberately smaller than
    Explore's 15 -- answering one question needs less room than the initial
    full exploration).
    """
    messages = list(messages)
    messages.append({"role": "user", "content": question})

    last_assistant_content: Optional[Sequence[Any]] = None
    exhausted = True

    for _ in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=QA_SYSTEM_PROMPT,
            tools=explorer.TOOLS,
            messages=messages,
        )

        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            last_assistant_content = response.content
            continue

        if response.stop_reason == "end_turn":
            messages.append({"role": "assistant", "content": response.content})
            last_assistant_content = response.content
            exhausted = False
            break

        tool_use_blocks = [b for b in response.content if _block_type(b) == "tool_use"]
        messages.append({"role": "assistant", "content": response.content})
        last_assistant_content = response.content

        if not tool_use_blocks:
            # No tool calls and not end_turn (e.g. max_tokens, refusal) -- stop here.
            exhausted = False
            break

        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": _block_attr(t, "id"),
                "content": explorer.execute_tool(
                    repo_root, _block_attr(t, "name"), _block_attr(t, "input")
                ),
            }
            for t in tool_use_blocks
        ]
        messages.append({"role": "user", "content": tool_results})

    answer = extract_answer_text(last_assistant_content)
    if not answer:
        if exhausted:
            answer = (
                f"(no final answer -- reached the {max_iterations}-iteration tool-loop cap "
                "while still exploring; try a narrower question)"
            )
        else:
            answer = "(the model returned no text answer for this question)"
    return answer, messages


def load_or_explore(
    client,
    model: str,
    repo_root: str,
    cache_dir: str,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Return (messages, from_cache). On a cache hit, prints a clear
    "reusing cached exploration" message and returns the cached transcript
    unmodified. On a miss, runs a fresh `explorer.run_explore_loop`
    (identical to what `generate` does) and best-effort saves it back to
    the cache -- so a later `generate` (or another `ask`/`chat`) against
    the same repo content + model gets a cache hit. A save failure is
    logged and never fatal, matching `pipeline.run_llm_phases`."""
    cached = cache_mod.load(cache_dir, repo_root, model)
    if cached is not None:
        print(
            "agent-demoforge: Explore-phase cache HIT -- reusing cached exploration of this "
            "repo instead of exploring again from scratch."
        )
        return cached, True

    print(
        "agent-demoforge: Explore-phase cache MISS -- exploring this repo fresh before "
        "answering (this exploration will be cached for future generate/ask/chat runs)."
    )
    messages = explorer.run_explore_loop(client, model, repo_root)
    try:
        cache_mod.save(cache_dir, repo_root, model, messages)
        print("agent-demoforge: saved this exploration to the Explore-phase cache.")
    except Exception as e:  # noqa: BLE001 - caching is a nice-to-have, never fatal
        print(f"agent-demoforge: WARNING: failed to write Explore-phase cache: {e}")
    return messages, False


def _print_safety_note() -> None:
    print(
        "agent-demoforge: read-only mode -- only the list_dir/read_file tools are available; "
        "no shell commands are ever executed, so (unlike `generate`) no --yes confirmation is needed."
    )


def speak_answer(answer: str, voice: Optional[str]) -> None:
    """Best-effort: speak `answer` aloud live via macOS `say` (see
    `tts.speak_live`). Truncates extremely long answers to
    `MAX_SPEAK_CHARS` before speaking so `--speak` can't hang reading out a
    huge answer -- the full, untruncated text is always printed to stdout
    regardless of this. Any failure (non-macOS, `say` missing, timeout) is
    reported, never raised."""
    text = answer
    if len(text) > MAX_SPEAK_CHARS:
        text = text[:MAX_SPEAK_CHARS] + "... (truncated for speech)"
    result = tts_mod.speak_live(text, voice=voice)
    if not result.ok:
        print(f"agent-demoforge: WARNING: --speak could not read the answer aloud ({result.error})")


def _prepare(source: str) -> str:
    print(f"agent-demoforge: preparing an isolated, read-only sandbox copy of '{source}'...")
    repo_root = sandbox.prepare_workdir(source)
    print(f"agent-demoforge: sandbox ready at: {repo_root}")
    print(
        "agent-demoforge: your original repository/URL will NOT be touched or written to; "
        "list_dir/read_file only ever read inside the temporary sandbox copy shown above."
    )
    _print_safety_note()
    return repo_root


def run_ask(
    source: str,
    question: str,
    model: str,
    voice: Optional[str] = None,
    speak: bool = False,
    cache_dir: Optional[str] = None,
) -> int:
    client, reason = check_credentials()
    if client is None:
        print(f"agent-demoforge: error: cannot answer questions without a live Anthropic API connection ({reason}).", file=sys.stderr)
        print(
            "agent-demoforge: unlike `generate`, `ask`/`chat` have no offline fallback for "
            "open-ended Q&A -- set ANTHROPIC_API_KEY and try again.",
            file=sys.stderr,
        )
        return 1

    try:
        repo_root = _prepare(source)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"agent-demoforge: error: {e}", file=sys.stderr)
        return 1

    cache_dir = cache_dir or os.path.join(os.getcwd(), cache_mod.CACHE_DIR_NAME)
    try:
        messages, _from_cache = load_or_explore(client, model, repo_root, cache_dir)
        print(f"agent-demoforge: answering (model: {model})...")
        answer, _messages = answer_question(client, model, repo_root, messages, question)
        print()
        print(answer)
        if speak:
            speak_answer(answer, voice)
        return 0
    except Exception as e:  # noqa: BLE001 - no offline fallback exists; report and exit clean
        print(f"agent-demoforge: error: failed to get an answer ({type(e).__name__}: {e})", file=sys.stderr)
        return 1
    finally:
        sandbox.cleanup_workdir(repo_root)


CHAT_EXIT_WORDS = {"exit", "quit"}


def run_chat(
    source: str,
    model: str,
    voice: Optional[str] = None,
    speak: bool = False,
    cache_dir: Optional[str] = None,
) -> int:
    client, reason = check_credentials()
    if client is None:
        print(f"agent-demoforge: error: cannot start a chat session without a live Anthropic API connection ({reason}).", file=sys.stderr)
        print(
            "agent-demoforge: unlike `generate`, `ask`/`chat` have no offline fallback for "
            "open-ended Q&A -- set ANTHROPIC_API_KEY and try again.",
            file=sys.stderr,
        )
        return 1

    try:
        repo_root = _prepare(source)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"agent-demoforge: error: {e}", file=sys.stderr)
        return 1

    cache_dir = cache_dir or os.path.join(os.getcwd(), cache_mod.CACHE_DIR_NAME)
    questions_answered = 0
    try:
        messages, _from_cache = load_or_explore(client, model, repo_root, cache_dir)

        print()
        print(f"agent-demoforge: chat about '{source}' (model: {model}). Type 'exit' or 'quit' (or Ctrl+D) to end.")
        print()

        while True:
            try:
                question = input("> ").strip()
            except EOFError:
                print()
                break

            if not question:
                continue
            if question.lower() in CHAT_EXIT_WORDS:
                break

            try:
                answer, messages = answer_question(client, model, repo_root, messages, question)
            except Exception as e:  # noqa: BLE001 - report this turn's failure, keep the session alive
                print(f"agent-demoforge: error answering that question ({type(e).__name__}: {e})")
                continue

            questions_answered += 1
            print()
            print(answer)
            print()
            if speak:
                speak_answer(answer, voice)

        print(f"agent-demoforge: chat session ended -- answered {questions_answered} question(s).")
        return 0
    finally:
        sandbox.cleanup_workdir(repo_root)
