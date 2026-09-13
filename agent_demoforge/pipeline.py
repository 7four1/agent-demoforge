"""Orchestrates the full agent-demoforge pipeline: explore -> script -> execute ->
narrate -> render -> video -> manifest.

Three-act structure: every generated DemoScript's beats are grouped into
(and, defensively, re-ordered into) the canonical section order
presentation -> code_walkthrough -> live_demo (restricted to whichever
sections were requested via --sections). A section-transition title card is
inserted at the start of each included section that actually has beats.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence, Tuple

from . import cache as cache_mod
from . import explorer, render, sandbox, scriptwriter, tts as tts_mod, video as video_mod
from .scriptwriter import CodeRef, DemoBeat, DemoScript, SECTION_DISPLAY_NAMES, SECTION_ORDER

DEFAULT_MODEL = "claude-opus-5"

WORDS_PER_SECOND = 2.5  # ~150 wpm, used to estimate duration when TTS is unavailable
MIN_ESTIMATED_DURATION = 2.0
FRAME_PADDING_SECONDS = 0.3
MIN_HOLD_SECONDS = 1.0

MAX_CODE_WALKTHROUGH_FILES = 3
CODE_EXCERPT_LINES = 16


@dataclass
class BeatRecord:
    index: int
    kind: str = "beat"  # "beat" | "section_title"
    section: str = "live_demo"
    narration: str = ""
    command: Optional[str] = None
    is_setup: bool = False
    bullets: Optional[List[str]] = None
    code_ref_path: Optional[str] = None
    code_ref_start: Optional[int] = None
    code_ref_end: Optional[int] = None
    code_ref_resolved: Optional[bool] = None
    exit_code: Optional[int] = None
    timed_out: bool = False
    skipped_reason: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    audio_ok: bool = False
    audio_path: Optional[str] = None
    audio_duration: float = 0.0
    audio_error: Optional[str] = None
    frame_paths: List[str] = field(default_factory=list)
    clip_path: Optional[str] = None


def _detect_console_script(repo_root: str) -> Optional[str]:
    pp = os.path.join(repo_root, "pyproject.toml")
    if os.path.isfile(pp):
        try:
            text = open(pp, encoding="utf-8", errors="replace").read()
        except OSError:
            return None
        m = re.search(r"\[project\.scripts\]\s*\n\s*([A-Za-z0-9_.-]+)\s*=", text)
        if m:
            return m.group(1)
    return None


def _extract_readme_pitch(repo_root: str) -> Optional[str]:
    """Pull a short, real pitch sentence out of the target repo's own
    README: the first non-empty, non-heading, non-code-fence paragraph
    after the title."""
    for name in ("README.md", "README.rst", "README.txt", "README"):
        path = os.path.join(repo_root, name)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except OSError:
                continue
            paragraph: List[str] = []
            in_code_fence = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code_fence = not in_code_fence
                    continue
                if in_code_fence:
                    continue
                if not stripped:
                    if paragraph:
                        break
                    continue
                if stripped.startswith("#") or stripped.startswith("!["):
                    continue
                paragraph.append(stripped)
            if paragraph:
                text = " ".join(paragraph)
                if len(text) > 240:
                    text = text[:237].rsplit(" ", 1)[0] + "..."
                return text
    return None


_IGNORED_SOURCE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".agent_demoforge_venv",
}


def _list_python_source_files(repo_root: str) -> List[str]:
    """Relative paths of real .py files in the repo, excluding venvs/caches
    and test files, sorted for determinism."""
    results = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(d for d in dirnames if d not in _IGNORED_SOURCE_DIRS)
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), repo_root)
            if "test" in rel.lower().split(os.sep):
                continue
            if os.path.basename(rel) in ("setup.py", "conftest.py"):
                continue
            results.append(rel)
    return sorted(results)


def _build_feature_bullets(
    repo_root: str,
    project_name: str,
    is_python: bool,
    console_script: Optional[str],
    has_readme: bool,
) -> List[str]:
    """Real, grounded bullet points for the presentation section's
    "key features" slide -- every claim here is backed by something
    actually observed on disk, never generic filler."""
    bullets: List[str] = []
    if console_script:
        bullets.append(f'Ships a "{console_script}" command-line entry point')

    py_files = _list_python_source_files(repo_root)
    if py_files:
        bullets.append(
            f"Implemented in {len(py_files)} Python source file"
            f"{'s' if len(py_files) != 1 else ''}"
        )

    has_tests = any(
        "test" in rel.lower().split(os.sep) or os.path.basename(rel).startswith("test_")
        for rel in _walk_all_files(repo_root)
    )
    if has_tests:
        bullets.append("Includes its own automated test suite")

    if is_python:
        req = os.path.join(repo_root, "requirements.txt")
        has_third_party_deps = False
        if os.path.isfile(req):
            try:
                has_third_party_deps = bool(open(req, encoding="utf-8", errors="replace").read().strip())
            except OSError:
                has_third_party_deps = True
        pp = os.path.join(repo_root, "pyproject.toml")
        if os.path.isfile(pp):
            try:
                text = open(pp, encoding="utf-8", errors="replace").read()
                m = re.search(r"dependencies\s*=\s*\[([^\]]*)\]", text)
                if m and m.group(1).strip():
                    has_third_party_deps = True
            except OSError:
                pass
        if not has_third_party_deps:
            bullets.append("Runs with zero third-party dependencies")

    if has_readme:
        bullets.append("Documents its own usage in a real README")

    return bullets[:5]


def _walk_all_files(repo_root: str) -> List[str]:
    results = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in _IGNORED_SOURCE_DIRS]
        for name in filenames:
            results.append(os.path.relpath(os.path.join(dirpath, name), repo_root))
    return results


_DEF_RE = re.compile(r"^\s*(def |class |async def )")


def _pick_code_excerpts(
    repo_root: str, max_files: int = MAX_CODE_WALKTHROUGH_FILES
) -> List[Tuple[str, int, int]]:
    """Heuristically pick up to `max_files` real (path, start_line,
    end_line) excerpts worth walking through: prefer conventionally-named
    entry-point/core files, and within each file prefer starting at its
    first function/class definition rather than its import block."""
    candidates = _list_python_source_files(repo_root)
    if not candidates:
        return []

    def score(rel: str) -> int:
        base = os.path.basename(rel).lower()
        depth = rel.count(os.sep)
        s = 0
        if base in ("cli.py", "__main__.py", "main.py"):
            s += 5
        if base in ("store.py", "core.py", "app.py", "server.py"):
            s += 3
        if base == "__init__.py":
            s -= 2
        s -= depth
        return s

    ranked = sorted(candidates, key=lambda rel: (-score(rel), rel))

    excerpts: List[Tuple[str, int, int]] = []
    for rel in ranked[:max_files]:
        full = os.path.join(repo_root, rel)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError:
            continue
        total = len(lines)
        if total == 0:
            continue
        start = 1
        for i, line in enumerate(lines):
            if _DEF_RE.match(line):
                start = i + 1
                break
        end = min(total, start + CODE_EXCERPT_LINES - 1)
        excerpts.append((rel, start, end))
    return excerpts


def build_fallback_script(
    repo_root: str,
    source: Optional[str] = None,
    author_name: Optional[str] = None,
    sections: Optional[Sequence[str]] = None,
) -> DemoScript:
    """A heuristic, non-LLM demo script used when the Anthropic API is
    unavailable, so the rest of the pipeline (execute/narrate/render/video)
    can still be exercised end-to-end. Clearly disclosed as a dry run.

    Produces beats for exactly the requested `sections` (default: all
    three), including real presentation bullets and real code_walkthrough
    excerpts read from the target repo's actual source -- never placeholder
    text.
    """
    requested = list(sections) if sections else list(SECTION_ORDER)
    included = [s for s in SECTION_ORDER if s in requested] or list(SECTION_ORDER)
    first_section = included[0]
    last_section = included[-1]

    name_source = source or repo_root
    project_name = os.path.basename(os.path.normpath(name_source).rstrip(os.sep)) or "this project"
    is_python = any(
        os.path.isfile(os.path.join(repo_root, f))
        for f in ("pyproject.toml", "setup.py", "requirements.txt")
    )
    console_script = _detect_console_script(repo_root)
    has_readme = os.path.isfile(os.path.join(repo_root, "README.md"))

    def intro_narration_for(section: str) -> str:
        if section == first_section and author_name:
            return (
                f"Hi, I'm putting together a quick demo of {project_name} on behalf of "
                f"{author_name}, without a live connection to the Claude API this time."
            )
        return (
            f"Here's a quick automated tour of {project_name}, put together without a "
            "live connection to the Claude API this time."
        )

    beats: List[DemoBeat] = []

    if "presentation" in included:
        beats.append(DemoBeat(section="presentation", narration=intro_narration_for("presentation")))
        pitch = _extract_readme_pitch(repo_root) or (
            f"{project_name} is a small, self-contained project meant to be run and "
            "inspected directly from the command line."
        )
        beats.append(DemoBeat(section="presentation", narration=pitch))
        bullets = _build_feature_bullets(repo_root, project_name, is_python, console_script, has_readme)
        if bullets:
            beats.append(
                DemoBeat(
                    section="presentation",
                    narration="Here's a quick look at how it's put together.",
                    bullets=bullets,
                )
            )
    elif first_section in ("code_walkthrough", "live_demo"):
        beats.append(DemoBeat(section=first_section, narration=intro_narration_for(first_section)))

    if "code_walkthrough" in included:
        excerpts = _pick_code_excerpts(repo_root)
        for rel_path, start, end in excerpts:
            beats.append(
                DemoBeat(
                    section="code_walkthrough",
                    narration=(
                        f"Let's take a look at {rel_path}, one of the real source files "
                        "this project is built from."
                    ),
                    code_ref=CodeRef(path=rel_path, start_line=start, end_line=end),
                )
            )

    if "live_demo" in included:
        if is_python:
            beats.append(
                DemoBeat(
                    section="live_demo",
                    narration="First, we install the project into a clean virtual environment.",
                    command="pip install -e .",
                    is_setup=True,
                )
            )
        if console_script:
            beats.append(
                DemoBeat(
                    section="live_demo",
                    narration="Let's add a task using the command line tool.",
                    command=f'{console_script} add "Write the quarterly report"',
                )
            )
            beats.append(
                DemoBeat(
                    section="live_demo",
                    narration="Now we add a second task.",
                    command=f'{console_script} add "Ship the agent-demoforge release"',
                )
            )
            beats.append(
                DemoBeat(
                    section="live_demo",
                    narration="Listing the tasks shows both of them, still open.",
                    command=f"{console_script} list",
                )
            )
            beats.append(
                DemoBeat(
                    section="live_demo",
                    narration="Marking the first one done updates its status.",
                    command=f"{console_script} done 1",
                )
            )
            beats.append(
                DemoBeat(
                    section="live_demo",
                    narration="Listing again shows that change reflected right away.",
                    command=f"{console_script} list",
                )
            )
        elif has_readme:
            beats.append(
                DemoBeat(
                    section="live_demo",
                    narration="Here's the project's own documentation.",
                    command="cat README.md",
                )
            )

    beats.append(
        DemoBeat(
            section=last_section,
            narration="And that wraps up this quick tour. Thanks for watching.",
        )
    )

    return DemoScript(title=f"{project_name} demo (offline dry run)", beats=beats)


def run_llm_phases(
    repo_root: str,
    model: str,
    author_name: Optional[str] = None,
    sections: Optional[Sequence[str]] = None,
    no_cache: bool = False,
    cache_dir: Optional[str] = None,
):
    """Attempt the live Explore + Script phases against the real Anthropic
    API. Returns (script_or_none, live: bool, reason_or_none).

    The Explore phase's transcript is best-effort cached under
    `cache_dir` (default: `.agent_demoforge_cache/` in the CWD), keyed by a
    fingerprint of the repo content + model. The Script phase always runs
    fresh, since --sections/--author-name/--voice can change independently
    of the repo's content.
    """
    try:
        import anthropic
    except ImportError:
        return None, False, "the 'anthropic' package is not installed"

    cache_dir = cache_dir or os.path.join(os.getcwd(), cache_mod.CACHE_DIR_NAME)

    try:
        client = anthropic.Anthropic()

        messages = None
        if not no_cache:
            messages = cache_mod.load(cache_dir, repo_root, model)
            if messages is not None:
                print(
                    "agent-demoforge: Explore-phase cache HIT -- reusing the cached transcript "
                    "and skipping the live tool-use loop."
                )

        if messages is None:
            if not no_cache:
                print("agent-demoforge: Explore-phase cache MISS -- running the live Explore tool-use loop.")
            messages = explorer.run_explore_loop(client, model, repo_root)
            if not no_cache:
                try:
                    cache_mod.save(cache_dir, repo_root, model, messages)
                except Exception as e:  # noqa: BLE001 - caching is a nice-to-have, never fatal
                    print(f"agent-demoforge: WARNING: failed to write Explore-phase cache: {e}")

        script = scriptwriter.write_script(
            client, model, messages, author_name=author_name, sections=sections
        )
        return script, True, None
    except Exception as e:  # noqa: BLE001 - any failure means: disclose and fall back
        return None, False, f"{type(e).__name__}: {e}"


def _estimate_duration(narration: str) -> float:
    words = max(1, len(narration.split()))
    return max(MIN_ESTIMATED_DURATION, words / WORDS_PER_SECOND)


def _print_cost_estimate(sections: Sequence[str]) -> None:
    explore_max = explorer.MAX_ITERATIONS
    low = 2  # best case: 1 Explore call that stops immediately + 1 Script call
    high = explore_max + 1  # worst case: uses every Explore iteration + 1 Script call
    note = ""
    if "code_walkthrough" not in sections:
        high = max(low, int(round(high * 0.7)))
        note = " (fewer are typically needed since --sections excludes code_walkthrough)"
    print()
    print(
        f"agent-demoforge: this run will make an estimated {low}-{high} Claude API calls "
        f"(messages.create/messages.parse) for the Explore+Script phases{note}. This is a "
        "rough call-count range based on the exploration iteration cap, not a token/cost "
        "prediction -- the exact number depends on how much exploration the model decides "
        "it needs."
    )


def _group_beats_by_section(beats: Sequence[DemoBeat], sections: Sequence[str]):
    groups = {s: [] for s in SECTION_ORDER}
    for b in beats:
        sec = b.section if b.section in groups and b.section in sections else None
        if sec is None:
            continue
        groups[sec].append(b)
    return groups


def _frame_entries_for_beat(
    beat: DemoBeat,
    record: BeatRecord,
    frames_dir: str,
    beat_index: int,
    beat_count: int,
    repo_root: str,
) -> List[tuple]:
    total_hold = record.audio_duration + FRAME_PADDING_SECONDS

    if beat.section == "presentation":
        frame1 = os.path.join(frames_dir, f"beat_{beat_index:02d}_a.png")
        render.render_slide_frame(frame1, title=beat.narration, bullets=beat.bullets)
        return [(frame1, max(MIN_HOLD_SECONDS, total_hold))]

    if beat.section == "code_walkthrough":
        frame1 = os.path.join(frames_dir, f"beat_{beat_index:02d}_a.png")
        if beat.code_ref:
            excerpt = explorer.read_code_excerpt(
                repo_root, beat.code_ref.path, beat.code_ref.start_line, beat.code_ref.end_line
            )
            if excerpt is not None:
                start, end, text = excerpt
                render.render_code_frame(frame1, beat.code_ref.path, start, end, text)
                record.code_ref_resolved = True
            else:
                print(
                    f"agent-demoforge: WARNING: code_walkthrough beat referenced "
                    f"'{beat.code_ref.path}' lines {beat.code_ref.start_line}-"
                    f"{beat.code_ref.end_line}, which could not be found on disk -- skipping "
                    "the code frame for this beat (narration still plays)."
                )
                render.render_code_missing_frame(frame1, beat.code_ref.path)
                record.code_ref_resolved = False
        else:
            render.render_slide_frame(frame1, title=beat.narration)
        return [(frame1, max(MIN_HOLD_SECONDS, total_hold))]

    # live_demo (and any unrecognized section, defensively treated the same way)
    if beat.command:
        frame1 = os.path.join(frames_dir, f"beat_{beat_index:02d}_a.png")
        render.render_frame(
            frame1,
            caption=beat.narration,
            beat_index=beat_index,
            beat_count=beat_count,
            command=beat.command,
            status="running..." if not record.skipped_reason else record.skipped_reason,
        )
        frame2 = os.path.join(frames_dir, f"beat_{beat_index:02d}_b.png")
        if record.skipped_reason:
            status2 = record.skipped_reason
        elif record.timed_out:
            status2 = f"timed out after {record.exit_code if record.exit_code is not None else '?'}"
        else:
            status2 = f"exit code {record.exit_code}"
        render.render_frame(
            frame2,
            caption=beat.narration,
            beat_index=beat_index,
            beat_count=beat_count,
            command=beat.command,
            status=status2,
            output_text=(record.stdout or "") + (("\n" + record.stderr) if record.stderr else ""),
        )
        hold1 = max(MIN_HOLD_SECONDS, total_hold * 0.35)
        hold2 = max(MIN_HOLD_SECONDS, total_hold - hold1)
        return [(frame1, hold1), (frame2, hold2)]

    frame1 = os.path.join(frames_dir, f"beat_{beat_index:02d}_a.png")
    render.render_frame(
        frame1,
        caption=beat.narration,
        beat_index=beat_index,
        beat_count=beat_count,
    )
    return [(frame1, max(MIN_HOLD_SECONDS, total_hold))]


def generate(
    source: Optional[str],
    out_dir: str,
    yes: bool = False,
    model: str = DEFAULT_MODEL,
    max_commands: int = 12,
    per_command_timeout: int = 90,
    max_wall_seconds: int = 480,
    voice: Optional[str] = "Samantha",
    author_name: Optional[str] = None,
    sections: Optional[Sequence[str]] = None,
    no_cache: bool = False,
) -> int:
    sections = [s for s in SECTION_ORDER if s in (sections or SECTION_ORDER)] or list(SECTION_ORDER)

    if not source:
        raise FileNotFoundError(
            "no repository given: pass a positional <repo-path-or-git-url>, or set "
            "AGENT_DEMOFORGE_REPO (directly or via a .env file)"
        )

    print(f"agent-demoforge: preparing an isolated sandbox copy of '{source}'...")
    repo_root = sandbox.prepare_workdir(source)
    print(f"agent-demoforge: sandbox ready at: {repo_root}")
    print("agent-demoforge: your original repository/URL will NOT be touched; every command below")
    print("           runs only inside the temporary sandbox copy shown above.")

    if not yes:
        print()
        print("WARNING: agent-demoforge is about to run shell commands from a generated demo script:")
        print(f"  - inside the temporary sandbox copy at: {repo_root}")
        print(f"  - never against your original path/URL: {source}")
        print(
            f"  - up to {max_commands} commands, each capped at {per_command_timeout}s, "
            f"total wall time capped at {max_wall_seconds}s"
        )
        try:
            reply = input("Proceed? [y/N] ").strip().lower()
        except EOFError:
            reply = "n"
        if reply not in ("y", "yes"):
            print("Aborted; no commands were run.")
            sandbox.cleanup_workdir(repo_root)
            return 1

    _print_cost_estimate(sections)

    script, llm_live, llm_reason = run_llm_phases(
        repo_root, model, author_name=author_name, sections=sections, no_cache=no_cache
    )
    if llm_live:
        print(f"agent-demoforge: explore+script phases completed live via the Anthropic API ({model}).")
    else:
        print(f"agent-demoforge: live Anthropic API phases unavailable ({llm_reason}).")
        print("agent-demoforge: falling back to an offline heuristic demo script (clearly marked in output).")
        script = build_fallback_script(repo_root, source=source, author_name=author_name, sections=sections)

    # Defensive: only keep beats for requested sections, even if the LLM
    # (or a future fallback change) produced something outside that set.
    dropped = [b for b in script.beats if b.section not in sections]
    if dropped:
        print(
            f"agent-demoforge: WARNING: dropping {len(dropped)} beat(s) whose section wasn't "
            f"requested via --sections ({', '.join(sections)})."
        )
    script.beats = [b for b in script.beats if b.section in sections]

    sb = sandbox.Sandbox(
        repo_root,
        per_command_timeout=per_command_timeout,
        max_commands=max_commands,
        max_wall_seconds=max_wall_seconds,
    )
    if sb.is_python_project():
        print("agent-demoforge: setting up a fresh virtualenv inside the sandbox...")
        try:
            sb.setup_venv()
        except Exception as e:  # noqa: BLE001
            print(f"agent-demoforge: WARNING: venv setup failed, running commands without one: {e}")

    backend = tts_mod.get_default_backend(voice=voice)
    tts_available = backend.is_available()
    if tts_available:
        print("agent-demoforge: narration audio will be synthesized via the macOS 'say' backend.")
    else:
        reason = getattr(backend, "reason", "unavailable")
        print(f"agent-demoforge: narration audio synthesis skipped ({reason}); using timed captions only.")

    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    audio_dir = os.path.join(out_dir, "audio")
    clips_dir = os.path.join(out_dir, "clips")
    for d in (frames_dir, audio_dir, clips_dir):
        os.makedirs(d, exist_ok=True)

    groups = _group_beats_by_section(script.beats, sections)
    sections_present = [s for s in SECTION_ORDER if s in sections and groups[s]]
    total_sections = len(sections_present)
    total_beats = sum(len(groups[s]) for s in sections_present)

    records: List[BeatRecord] = []
    clip_paths: List[str] = []
    chapters: List[Tuple[float, str]] = []
    running_time = 0.0
    global_index = 0
    beat_number = 0

    def synthesize(text: str, path: str) -> Tuple[bool, float, Optional[str], Optional[str]]:
        if tts_available:
            result = backend.synthesize(text, path)
            if result.ok:
                return True, result.duration_seconds, result.path, None
            return False, _estimate_duration(text), None, result.error
        return False, _estimate_duration(text), None, getattr(backend, "reason", "TTS unavailable")

    for sec_idx, sec in enumerate(sections_present, start=1):
        display_name = SECTION_DISPLAY_NAMES[sec]
        global_index += 1
        title_text = f"Part {sec_idx} of {total_sections} -- {display_name}"
        spoken_text = f"Part {sec_idx} of {total_sections}. {display_name}."
        print(f"agent-demoforge: section {sec_idx}/{total_sections}: {display_name}")

        chapters.append((running_time, display_name))

        rec = BeatRecord(index=global_index, kind="section_title", section=sec, narration=title_text)
        audio_path = os.path.join(audio_dir, f"section_{sec_idx:02d}.aiff")
        ok, duration, real_path, err = synthesize(spoken_text, audio_path)
        rec.audio_ok, rec.audio_duration, rec.audio_path, rec.audio_error = ok, duration, real_path, err

        frame_path = os.path.join(frames_dir, f"section_{sec_idx:02d}_title.png")
        render.render_section_card(frame_path, sec_idx, total_sections, display_name)
        rec.frame_paths = [frame_path]

        audio_for_mux = rec.audio_path if rec.audio_ok else None
        if not audio_for_mux:
            audio_for_mux = os.path.join(audio_dir, f"section_{sec_idx:02d}_silent.m4a")
            video_mod.render_silent_audio(audio_for_mux, rec.audio_duration)

        clip_path = os.path.join(clips_dir, f"section_{sec_idx:02d}.mp4")
        hold = max(MIN_HOLD_SECONDS, rec.audio_duration + FRAME_PADDING_SECONDS)
        video_mod.build_beat_clip([(frame_path, hold)], audio_for_mux, clip_path)
        rec.clip_path = clip_path
        clip_paths.append(clip_path)
        running_time += max(0.5, rec.audio_duration)
        records.append(rec)

        for beat in groups[sec]:
            beat_number += 1
            global_index += 1
            print(f"agent-demoforge: beat {beat_number}/{total_beats} [{sec}]: {beat.narration!r}")
            record = BeatRecord(
                index=global_index,
                kind="beat",
                section=sec,
                narration=beat.narration,
                command=beat.command,
                is_setup=beat.is_setup,
                bullets=beat.bullets,
                code_ref_path=beat.code_ref.path if beat.code_ref else None,
                code_ref_start=beat.code_ref.start_line if beat.code_ref else None,
                code_ref_end=beat.code_ref.end_line if beat.code_ref else None,
            )

            # --- Narrate ---
            audio_path = os.path.join(audio_dir, f"beat_{global_index:02d}.aiff")
            ok, duration, real_path, err = synthesize(beat.narration, audio_path)
            record.audio_ok, record.audio_duration, record.audio_path, record.audio_error = (
                ok,
                duration,
                real_path,
                err,
            )

            # --- Execute ---
            if beat.command:
                try:
                    cmd_result = sb.run(beat.command)
                    record.exit_code = cmd_result.exit_code
                    record.timed_out = cmd_result.timed_out
                    record.stdout = cmd_result.stdout
                    record.stderr = cmd_result.stderr
                except sandbox.CommandBudgetExceeded as e:
                    record.skipped_reason = f"skipped ({e})"
                    record.stderr = record.skipped_reason

            # --- Render frames ---
            frame_entries = _frame_entries_for_beat(
                beat, record, frames_dir, global_index, total_beats, repo_root
            )
            record.frame_paths = [p for p, _ in frame_entries]

            # --- Assemble per-beat clip ---
            audio_for_mux = record.audio_path if record.audio_ok else None
            if not audio_for_mux:
                audio_for_mux = os.path.join(audio_dir, f"beat_{global_index:02d}_silent.m4a")
                video_mod.render_silent_audio(audio_for_mux, record.audio_duration)

            clip_path = os.path.join(clips_dir, f"beat_{global_index:02d}.mp4")
            video_mod.build_beat_clip(frame_entries, audio_for_mux, clip_path)
            record.clip_path = clip_path

            records.append(record)
            clip_paths.append(clip_path)
            running_time += max(0.5, record.audio_duration)

    print("agent-demoforge: concatenating beat clips into demo.mp4 ...")
    demo_mp4 = os.path.join(out_dir, "demo.mp4")
    chapters_added = False
    try:
        video_mod.concat_clips(clip_paths, demo_mp4, chapters=chapters, chapter_total_duration=running_time)
        chapters_added = True
    except Exception as e:  # noqa: BLE001 - chapters are a nice-to-have, never fatal
        print(f"agent-demoforge: WARNING: chapter-tagged concat failed ({e}); retrying without chapters.")
        video_mod.concat_clips(clip_paths, demo_mp4)
        chapters_added = False

    print("agent-demoforge: rendering demo.gif preview ...")
    demo_gif = os.path.join(out_dir, "demo.gif")
    video_mod.make_gif(clip_paths, demo_gif)

    narration_md = os.path.join(out_dir, "narration_script.md")
    _write_narration_markdown(script.title, records, narration_md, llm_live, llm_reason, tts_available, backend)

    manifest_path = os.path.join(out_dir, "manifest.json")
    _write_manifest(
        script.title, records, manifest_path, source, repo_root, model,
        llm_live, llm_reason, tts_available, backend, demo_mp4, demo_gif,
        voice=voice, author_name=author_name, sections=sections,
        chapters=chapters, chapters_added=chapters_added,
    )

    print()
    print(f"agent-demoforge: done. Output written to: {os.path.abspath(out_dir)}")
    print(f"  - {demo_mp4}")
    print(f"  - {demo_gif}")
    print(f"  - {narration_md}")
    print(f"  - {manifest_path}")

    return 0


def _write_narration_markdown(
    title: str,
    records: List[BeatRecord],
    out_path: str,
    llm_live: bool,
    llm_reason: Optional[str],
    tts_available: bool,
    backend,
) -> None:
    lines = [f"# {title}", ""]
    if llm_live:
        lines.append("_Explore/Script phase: live Anthropic API call._")
    else:
        lines.append(f"_Explore/Script phase: offline dry-run stub ({llm_reason})._")
    if tts_available:
        lines.append("_Narration audio: synthesized via the macOS `say` backend._")
    else:
        lines.append(
            f"_Narration audio: SKIPPED ({getattr(backend, 'reason', 'unavailable')}); "
            "durations below are estimated from word count._"
        )
    lines.append("")

    for rec in records:
        if rec.kind == "section_title":
            lines.append(f"# {rec.narration}")
            lines.append("")
            continue

        lines.append(f"## Beat {rec.index} _({rec.section})_")
        lines.append("")
        lines.append(f"**Narration:** {rec.narration}")
        lines.append("")
        if rec.bullets:
            for b in rec.bullets:
                lines.append(f"- {b}")
            lines.append("")
        if rec.code_ref_path:
            resolved = "real file content rendered" if rec.code_ref_resolved else "SKIPPED (not found on disk)"
            lines.append(
                f"**Code reference:** `{rec.code_ref_path}` lines "
                f"{rec.code_ref_start}-{rec.code_ref_end} ({resolved})"
            )
            lines.append("")
        if rec.command:
            tag = " _(setup)_" if rec.is_setup else ""
            lines.append(f"**Command:** `{rec.command}`{tag}")
            lines.append("")
            if rec.skipped_reason:
                lines.append(f"_{rec.skipped_reason}_")
            else:
                status = "timed out" if rec.timed_out else f"exit code {rec.exit_code}"
                lines.append(f"Status: {status}")
                if rec.stdout.strip():
                    lines.append("")
                    lines.append("```")
                    lines.append(rec.stdout.strip()[:2000])
                    lines.append("```")
                if rec.stderr.strip():
                    lines.append("")
                    lines.append("stderr:")
                    lines.append("```")
                    lines.append(rec.stderr.strip()[:2000])
                    lines.append("```")
            lines.append("")
        audio_note = "" if rec.audio_ok else " (estimated -- narration audio unavailable)"
        lines.append(f"Audio duration: {rec.audio_duration:.2f}s{audio_note}")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_manifest(
    title: str,
    records: List[BeatRecord],
    out_path: str,
    source: str,
    repo_root: str,
    model: str,
    llm_live: bool,
    llm_reason: Optional[str],
    tts_available: bool,
    backend,
    demo_mp4: str,
    demo_gif: str,
    voice: Optional[str] = None,
    author_name: Optional[str] = None,
    sections: Optional[Sequence[str]] = None,
    chapters: Optional[List[Tuple[float, str]]] = None,
    chapters_added: bool = False,
) -> None:
    manifest = {
        "title": title,
        "source": source,
        "sandbox_repo_root": repo_root,
        "model": model,
        "author_name": author_name,
        "sections": list(sections or []),
        "llm_phase": {"live": llm_live, "reason": llm_reason},
        "tts": {
            "available": tts_available,
            "backend": type(backend).__name__,
            "voice": voice or getattr(backend, "voice", None),
            "reason": None if tts_available else getattr(backend, "reason", None),
        },
        "chapters": {
            "added": chapters_added,
            "marks": [{"start_seconds": round(t, 2), "title": name} for t, name in (chapters or [])],
        },
        "beats": [
            {
                **asdict(rec),
                "frame_paths": [os.path.abspath(p) for p in rec.frame_paths],
                "clip_path": os.path.abspath(rec.clip_path) if rec.clip_path else None,
                "audio_path": os.path.abspath(rec.audio_path) if rec.audio_path else None,
            }
            for rec in records
        ],
        "total_audio_duration_seconds": round(sum(r.audio_duration for r in records), 2),
        "outputs": {
            "demo_mp4": os.path.abspath(demo_mp4),
            "demo_gif": os.path.abspath(demo_gif),
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
