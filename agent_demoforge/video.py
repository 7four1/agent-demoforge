"""Video assembly via imageio-ffmpeg's bundled, real, pip-installable ffmpeg
binary -- no system ffmpeg install required.

Recipe (reused identically for every clip so the final concat step works
cleanly): for each beat, turn its frame image(s) -- held for durations that
sum to at least the narration audio's real duration -- into a silent clip
via the ffmpeg concat demuxer, then mux on the beat's narration audio (real
`say` output, or a generated silent track if narration was unavailable for
that beat) re-encoded as H.264/AAC. All per-beat clips share that same
encoding, so the final concat + one more clean re-encode reliably produces
one playable demo.mp4. A separate concat pass over the same per-beat clips
produces demo.gif for embedding in a README.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Optional, Sequence, Tuple

import imageio_ffmpeg

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FPS = 25

X264_AAC_ARGS = [
    "-vf",
    f"fps={FPS},scale={VIDEO_WIDTH}:{VIDEO_HEIGHT},format=yuv420p",
    "-c:v",
    "libx264",
    "-preset",
    "veryfast",
    "-crf",
    "23",
]


class FfmpegError(RuntimeError):
    pass


def get_ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args: Sequence[str], timeout: int = 180) -> str:
    exe = get_ffmpeg_exe()
    cmd = [exe, "-y", "-loglevel", "error"] + list(args)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise FfmpegError(
            f"ffmpeg failed (exit {proc.returncode}): {' '.join(cmd)}\n{proc.stderr}"
        )
    return proc.stderr


def probe_streams(path: str, timeout: int = 30) -> str:
    """Run `ffmpeg -i <path>` and return its stderr, which carries the
    stream-info banner (codec, duration, resolution, sample rate, ...).
    `ffmpeg -i` with no output always exits non-zero -- that's expected."""
    exe = get_ffmpeg_exe()
    proc = subprocess.run([exe, "-i", path], capture_output=True, text=True, timeout=timeout)
    return proc.stderr


def _write_concat_list(list_path: str, entries: Sequence[Tuple[str, float]]) -> None:
    """Write an ffmpeg concat-demuxer list for a sequence of (image_path,
    hold_seconds). The concat demuxer requires the last file to be repeated
    once more without a duration line, or its final segment gets dropped."""
    lines = []
    for path, duration in entries:
        abs_path = os.path.abspath(path)
        lines.append(f"file '{abs_path}'")
        lines.append(f"duration {duration:.3f}")
    if entries:
        lines.append(f"file '{os.path.abspath(entries[-1][0])}'")
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def render_silent_audio(out_path: str, duration_seconds: float) -> str:
    """Generate a silent AAC audio track of the given duration, used when
    real narration audio is unavailable for a beat (degraded mode) so every
    clip has a consistent audio+video stream layout for concatenation."""
    duration_seconds = max(0.5, duration_seconds)
    run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo",
            "-t",
            f"{duration_seconds:.3f}",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            out_path,
        ]
    )
    return out_path


def build_beat_clip(
    frame_entries: Sequence[Tuple[str, float]],
    audio_path: str,
    out_path: str,
) -> str:
    """Build one beat's video clip: concat its frame(s) (held for the given
    durations) into video, muxed with `audio_path`, trimmed to the shorter
    of the two (frame durations are chosen upstream to sum to >= the audio
    duration, so in practice this trims trailing frame padding)."""
    list_path = out_path + ".concat.txt"
    _write_concat_list(list_path, frame_entries)
    try:
        run_ffmpeg(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-i",
                audio_path,
                *X264_AAC_ARGS,
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-shortest",
                out_path,
            ]
        )
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)
    return out_path


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def _parse_duration_seconds(ffmpeg_banner: str) -> Optional[float]:
    m = _DURATION_RE.search(ffmpeg_banner)
    if not m:
        return None
    h, mm, s = m.groups()
    return int(h) * 3600 + int(mm) * 60 + float(s)


def write_chapters_metadata(
    meta_path: str, chapters: Sequence[Tuple[float, str]], total_duration: float
) -> str:
    """Write an ffmpeg FFMETADATA1 file with one [CHAPTER] block per
    (start_seconds, title) entry in `chapters` (must be sorted ascending by
    start time). Each chapter's END is the next chapter's START, or
    `total_duration` for the last one."""
    lines = [";FFMETADATA1"]
    for i, (start, title) in enumerate(chapters):
        end = chapters[i + 1][0] if i + 1 < len(chapters) else total_duration
        start_ms = max(0, int(round(start * 1000)))
        end_ms = int(round(end * 1000))
        if end_ms <= start_ms:
            end_ms = start_ms + 1
        safe_title = (
            (title or "")
            .replace("\\", "\\\\")
            .replace("=", "\\=")
            .replace(";", "\\;")
            .replace("#", "\\#")
            .replace("\n", " ")
        )
        lines.append("[CHAPTER]")
        lines.append("TIMEBASE=1/1000")
        lines.append(f"START={start_ms}")
        lines.append(f"END={end_ms}")
        lines.append(f"title={safe_title}")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return meta_path


def add_chapters(
    input_path: str,
    output_path: str,
    chapters: Sequence[Tuple[float, str]],
    total_duration: float,
) -> bool:
    """Remux `input_path` to `output_path`, attaching chapter markers via an
    FFMETADATA sidecar file (the documented ffmpeg recipe: pass it as a
    second input, then `-map_metadata 1 -map_chapters 1`). Stream-copies
    (no re-encode) since only container metadata changes. Returns True on
    success, False on any failure -- callers should fall back to the
    un-chaptered file rather than fail the whole pipeline over this
    nice-to-have."""
    meta_path = output_path + ".chapters.txt"
    write_chapters_metadata(meta_path, chapters, total_duration)
    try:
        run_ffmpeg(
            [
                "-i",
                input_path,
                "-i",
                meta_path,
                "-map_metadata",
                "1",
                "-map_chapters",
                "1",
                "-codec",
                "copy",
                output_path,
            ]
        )
        return True
    except FfmpegError:
        return False
    finally:
        if os.path.exists(meta_path):
            os.remove(meta_path)


def concat_clips(
    clip_paths: Sequence[str],
    out_path: str,
    chapters: Optional[Sequence[Tuple[float, str]]] = None,
    chapter_total_duration: Optional[float] = None,
) -> str:
    """Concatenate uniformly-encoded per-beat clips into one final mp4.
    Re-encodes (rather than stream-copying) for robustness against any
    minor timestamp/keyframe inconsistency between clips.

    If `chapters` (a list of (start_seconds, title), ascending) is given,
    chapter metadata marking each section's start is attached as a second,
    stream-copy remux pass -- best effort: if that pass fails for any
    reason, the plain (un-chaptered) file is used instead so a nice-to-have
    never breaks the base video.
    """
    list_path = out_path + ".concat.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")

    target = out_path + ".prechapters.mp4" if chapters else out_path
    try:
        run_ffmpeg(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                *X264_AAC_ARGS,
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                target,
            ]
        )
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)

    if chapters:
        total = chapter_total_duration
        if total is None:
            total = _parse_duration_seconds(probe_streams(target)) or 0.0
        ok = add_chapters(target, out_path, chapters, total)
        if not ok:
            shutil.copyfile(target, out_path)
        try:
            os.remove(target)
        except OSError:
            pass

    return out_path


def make_gif(clip_paths: Sequence[str], out_path: str, fps: float = 1.5, width: int = 800) -> str:
    """Render a lightweight, README-embeddable silent GIF preview from the
    same per-beat clips."""
    list_path = out_path + ".concat.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    try:
        run_ffmpeg(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-vf",
                f"fps={fps},scale={width}:-1:flags=lanczos",
                "-loop",
                "0",
                out_path,
            ]
        )
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)
    return out_path
