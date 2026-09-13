"""Orchestrates the full agent-demoforge pipeline: explore -> script -> execute ->
narrate -> render -> video -> manifest."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from . import explorer, render, sandbox, scriptwriter, tts as tts_mod, video as video_mod
from .scriptwriter import DemoBeat, DemoScript

DEFAULT_MODEL = "claude-opus-5"

WORDS_PER_SECOND = 2.5  # ~150 wpm, used to estimate duration when TTS is unavailable
MIN_ESTIMATED_DURATION = 2.0
FRAME_PADDING_SECONDS = 0.3
MIN_HOLD_SECONDS = 1.0


@dataclass
class BeatRecord:
    index: int
    narration: str
    command: Optional[str]
    is_setup: bool
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


def build_fallback_script(
    repo_root: str, source: Optional[str] = None, author_name: Optional[str] = None
) -> DemoScript:
    """A heuristic, non-LLM demo script used when the Anthropic API is
    unavailable, so the rest of the pipeline (execute/narrate/render/video)
    can still be exercised end-to-end. Clearly disclosed as a dry run."""
    name_source = source or repo_root
    project_name = os.path.basename(os.path.normpath(name_source).rstrip(os.sep)) or "this project"
    is_python = any(
        os.path.isfile(os.path.join(repo_root, f))
        for f in ("pyproject.toml", "setup.py", "requirements.txt")
    )
    console_script = _detect_console_script(repo_root)
    has_readme = os.path.isfile(os.path.join(repo_root, "README.md"))

    if author_name:
        intro_narration = (
            f"Hi, I'm putting together a quick demo of {project_name} on behalf of "
            f"{author_name}, without a live connection to the Claude API this time."
        )
    else:
        intro_narration = (
            f"Here's a quick automated tour of {project_name}, put together without a "
            "live connection to the Claude API this time."
        )

    beats: List[DemoBeat] = [DemoBeat(narration=intro_narration)]

    if is_python:
        beats.append(
            DemoBeat(
                narration="First, we install the project into a clean virtual environment.",
                command="pip install -e .",
                is_setup=True,
            )
        )

    if console_script:
        beats.append(
            DemoBeat(
                narration="Let's add a task using the command line tool.",
                command=f'{console_script} add "Write the quarterly report"',
            )
        )
        beats.append(
            DemoBeat(
                narration="Now we add a second task.",
                command=f'{console_script} add "Ship the agent-demoforge release"',
            )
        )
        beats.append(
            DemoBeat(
                narration="Listing the tasks shows both of them, still open.",
                command=f"{console_script} list",
            )
        )
        beats.append(
            DemoBeat(
                narration="Marking the first one done updates its status.",
                command=f"{console_script} done 1",
            )
        )
        beats.append(
            DemoBeat(
                narration="Listing again shows that change reflected right away.",
                command=f"{console_script} list",
            )
        )
    elif has_readme:
        beats.append(
            DemoBeat(
                narration="Here's the project's own documentation.",
                command="cat README.md",
            )
        )

    beats.append(DemoBeat(narration="And that wraps up this quick tour. Thanks for watching."))

    return DemoScript(title=f"{project_name} demo (offline dry run)", beats=beats)


def run_llm_phases(repo_root: str, model: str, author_name: Optional[str] = None):
    """Attempt the live Explore + Script phases against the real Anthropic
    API. Returns (script_or_none, live: bool, reason_or_none)."""
    try:
        import anthropic
    except ImportError:
        return None, False, "the 'anthropic' package is not installed"

    try:
        client = anthropic.Anthropic()
        messages = explorer.run_explore_loop(client, model, repo_root)
        script = scriptwriter.write_script(client, model, messages, author_name=author_name)
        return script, True, None
    except Exception as e:  # noqa: BLE001 - any failure means: disclose and fall back
        return None, False, f"{type(e).__name__}: {e}"


def _estimate_duration(narration: str) -> float:
    words = max(1, len(narration.split()))
    return max(MIN_ESTIMATED_DURATION, words / WORDS_PER_SECOND)


def _frame_entries_for_beat(
    beat: DemoBeat,
    record: BeatRecord,
    frames_dir: str,
    beat_index: int,
    beat_count: int,
) -> List[tuple]:
    total_hold = record.audio_duration + FRAME_PADDING_SECONDS

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
    source: str,
    out_dir: str,
    yes: bool = False,
    model: str = DEFAULT_MODEL,
    max_commands: int = 12,
    per_command_timeout: int = 90,
    max_wall_seconds: int = 480,
    voice: Optional[str] = "Daniel",
    author_name: Optional[str] = None,
) -> int:
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

    script, llm_live, llm_reason = run_llm_phases(repo_root, model, author_name=author_name)
    if llm_live:
        print(f"agent-demoforge: explore+script phases completed live via the Anthropic API ({model}).")
    else:
        print(f"agent-demoforge: live Anthropic API phases unavailable ({llm_reason}).")
        print("agent-demoforge: falling back to an offline heuristic demo script (clearly marked in output).")
        script = build_fallback_script(repo_root, source=source, author_name=author_name)

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

    beat_count = len(script.beats)
    records: List[BeatRecord] = []

    for i, beat in enumerate(script.beats, start=1):
        print(f"agent-demoforge: beat {i}/{beat_count}: {beat.narration!r}")
        record = BeatRecord(
            index=i, narration=beat.narration, command=beat.command, is_setup=beat.is_setup
        )

        # --- Narrate ---
        if tts_available:
            audio_path = os.path.join(audio_dir, f"beat_{i:02d}.aiff")
            result = backend.synthesize(beat.narration, audio_path)
            record.audio_ok = result.ok
            record.audio_duration = result.duration_seconds
            record.audio_path = result.path if result.ok else None
            record.audio_error = result.error
        else:
            record.audio_error = getattr(backend, "reason", "TTS unavailable")

        if not record.audio_ok:
            record.audio_duration = _estimate_duration(beat.narration)

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
        frame_entries = _frame_entries_for_beat(beat, record, frames_dir, i, beat_count)
        record.frame_paths = [p for p, _ in frame_entries]

        # --- Assemble per-beat clip ---
        audio_for_mux = record.audio_path if record.audio_ok else None
        if not audio_for_mux:
            audio_for_mux = os.path.join(audio_dir, f"beat_{i:02d}_silent.m4a")
            video_mod.render_silent_audio(audio_for_mux, record.audio_duration)

        clip_path = os.path.join(clips_dir, f"beat_{i:02d}.mp4")
        video_mod.build_beat_clip(frame_entries, audio_for_mux, clip_path)
        record.clip_path = clip_path

        records.append(record)

    print("agent-demoforge: concatenating beat clips into demo.mp4 ...")
    demo_mp4 = os.path.join(out_dir, "demo.mp4")
    video_mod.concat_clips([r.clip_path for r in records], demo_mp4)

    print("agent-demoforge: rendering demo.gif preview ...")
    demo_gif = os.path.join(out_dir, "demo.gif")
    video_mod.make_gif([r.clip_path for r in records], demo_gif)

    narration_md = os.path.join(out_dir, "narration_script.md")
    _write_narration_markdown(
        script, records, narration_md, llm_live, llm_reason, tts_available, backend
    )

    manifest_path = os.path.join(out_dir, "manifest.json")
    _write_manifest(
        script, records, manifest_path, source, repo_root, model,
        llm_live, llm_reason, tts_available, backend, demo_mp4, demo_gif,
        voice=voice, author_name=author_name,
    )

    print()
    print(f"agent-demoforge: done. Output written to: {os.path.abspath(out_dir)}")
    print(f"  - {demo_mp4}")
    print(f"  - {demo_gif}")
    print(f"  - {narration_md}")
    print(f"  - {manifest_path}")

    return 0


def _write_narration_markdown(
    script: DemoScript,
    records: List[BeatRecord],
    out_path: str,
    llm_live: bool,
    llm_reason: Optional[str],
    tts_available: bool,
    backend,
) -> None:
    lines = [f"# {script.title}", ""]
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

    for beat, rec in zip(script.beats, records):
        lines.append(f"## Beat {rec.index}")
        lines.append("")
        lines.append(f"**Narration:** {beat.narration}")
        lines.append("")
        if beat.command:
            tag = " _(setup)_" if beat.is_setup else ""
            lines.append(f"**Command:** `{beat.command}`{tag}")
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
    script: DemoScript,
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
) -> None:
    manifest = {
        "title": script.title,
        "source": source,
        "sandbox_repo_root": repo_root,
        "model": model,
        "author_name": author_name,
        "llm_phase": {"live": llm_live, "reason": llm_reason},
        "tts": {
            "available": tts_available,
            "backend": type(backend).__name__,
            "voice": voice or getattr(backend, "voice", None),
            "reason": None if tts_available else getattr(backend, "reason", None),
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
