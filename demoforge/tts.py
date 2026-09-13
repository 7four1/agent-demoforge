"""Pluggable text-to-speech backends.

Ships one real, verified backend: macOS `say`, writing AIFF audio. The
`TTSBackend` interface is small enough that a non-macOS user can add e.g. a
Piper or ElevenLabs backend later without touching the rest of the
pipeline -- only `tts.get_default_backend()` and the pipeline's call site
would need to change.

AIFF duration is read by parsing the file's `COMM` chunk directly (channel
count, sample-frame count, sample rate as an 80-bit IEEE-754 extended
float) rather than via the standard library's `aifc` module: `aifc` was
removed in Python 3.13 (PEP 594), so relying on it would break on current
Python. The parsing below has no dependencies and is fully reliable for
the AIFF files `say` produces.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TTSResult:
    path: Optional[str]
    duration_seconds: float
    ok: bool
    error: Optional[str] = None


class TTSBackend(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this backend can actually synthesize audio right now."""

    @abstractmethod
    def synthesize(self, text: str, out_path: str) -> TTSResult:
        """Synthesize `text` to a real audio file at `out_path`."""


def _read_ieee_extended(chunk: bytes) -> float:
    """Decode a big-endian 80-bit IEEE-754 extended precision float, as
    used by the AIFF COMM chunk's sampleRate field."""
    if len(chunk) < 10:
        raise ValueError("IEEE extended float chunk must be 10 bytes")
    expon = struct.unpack(">H", chunk[0:2])[0]
    sign = -1.0 if expon & 0x8000 else 1.0
    expon &= 0x7FFF
    himant, lomant = struct.unpack(">II", chunk[2:10])
    if expon == 0 and himant == 0 and lomant == 0:
        return 0.0
    if expon == 0x7FFF:  # infinity or NaN
        return float("inf")
    expon -= 16383
    f = (himant * 4294967296.0 + lomant) * (2.0 ** (expon - 63))
    return sign * f


def read_aiff_duration(path: str) -> float:
    """Return the duration in seconds of an AIFF file by parsing its COMM
    chunk directly (numSampleFrames / sampleRate)."""
    with open(path, "rb") as f:
        data = f.read()

    if len(data) < 12 or data[0:4] != b"FORM" or data[8:12] not in (b"AIFF", b"AIFC"):
        raise ValueError(f"'{path}' is not a valid AIFF/AIFC file")

    pos = 12
    num_frames = None
    sample_rate = None
    n = len(data)
    while pos + 8 <= n:
        chunk_id = data[pos : pos + 4]
        chunk_size = struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        body_start = pos + 8
        if chunk_id == b"COMM":
            if body_start + 18 > n:
                raise ValueError("truncated COMM chunk")
            num_frames = struct.unpack(">I", data[body_start + 2 : body_start + 6])[0]
            sample_rate = _read_ieee_extended(data[body_start + 8 : body_start + 18])
            break
        # Chunks are word (2-byte) aligned; pad by 1 if chunk_size is odd.
        pos = body_start + chunk_size + (chunk_size % 2)

    if num_frames is None or not sample_rate:
        raise ValueError(f"no COMM chunk found in '{path}'")
    return num_frames / sample_rate


class MacSayTTSBackend(TTSBackend):
    """Uses the macOS `say` command to synthesize real speech audio."""

    def __init__(self, voice: Optional[str] = None, sample_rate: int = 22050):
        self.voice = voice
        self.sample_rate = sample_rate

    def is_available(self) -> bool:
        return shutil.which("say") is not None

    def _run_say(self, cmd, timeout=60):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def synthesize(self, text: str, out_path: str) -> TTSResult:
        if not self.is_available():
            return TTSResult(
                path=None,
                duration_seconds=0.0,
                ok=False,
                error="the 'say' command is not available (macOS-only backend)",
            )
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

        voice_args = ["-v", self.voice] if self.voice else []

        # Prefer an explicit raw PCM data-format for a predictable, easy-to-parse
        # AIFF. Some macOS/`say` builds reject --data-format entirely (observed:
        # "Opening output file failed: fmt?" regardless of the value passed) --
        # when that happens, fall back to `say`'s own default output format,
        # which is still a real, valid AIFF-C file our AIFF/AIFC duration
        # reader below handles just as well.
        primary_cmd = ["say", *voice_args, "-o", out_path, f"--data-format=LEF32@{self.sample_rate}", text]
        proc = self._run_say(primary_cmd)

        succeeded = (
            proc is not None
            and proc.returncode == 0
            and os.path.isfile(out_path)
            and os.path.getsize(out_path) > 0
        )

        if not succeeded:
            fallback_cmd = ["say", *voice_args, "-o", out_path, text]
            proc = self._run_say(fallback_cmd)
            succeeded = (
                proc is not None
                and proc.returncode == 0
                and os.path.isfile(out_path)
                and os.path.getsize(out_path) > 0
            )

        if not succeeded:
            if proc is None:
                return TTSResult(path=None, duration_seconds=0.0, ok=False, error="'say' timed out")
            return TTSResult(
                path=None,
                duration_seconds=0.0,
                ok=False,
                error=f"'say' failed (exit {proc.returncode}): {proc.stderr.strip()}",
            )

        try:
            duration = read_aiff_duration(out_path)
        except Exception as e:  # noqa: BLE001 - report, don't crash the pipeline
            return TTSResult(
                path=out_path,
                duration_seconds=0.0,
                ok=False,
                error=f"synthesized audio but could not read its duration: {e}",
            )

        return TTSResult(path=out_path, duration_seconds=duration, ok=True)


class NullTTSBackend(TTSBackend):
    """Fallback backend for platforms with no TTS engine wired up. Always
    reports unavailable so the pipeline degrades gracefully instead of
    crashing."""

    def __init__(self, reason: str = "no text-to-speech backend is available on this platform"):
        self.reason = reason

    def is_available(self) -> bool:
        return False

    def synthesize(self, text: str, out_path: str) -> TTSResult:
        return TTSResult(path=None, duration_seconds=0.0, ok=False, error=self.reason)


def get_default_backend(voice: Optional[str] = None) -> TTSBackend:
    """Pick the best backend available in the current environment."""
    import sys

    if sys.platform == "darwin":
        backend = MacSayTTSBackend(voice=voice)
        if backend.is_available():
            return backend
        return NullTTSBackend("this Mac has no 'say' command on PATH")
    return NullTTSBackend(
        f"no TTS backend is implemented for platform '{sys.platform}' yet "
        "(only macOS 'say' ships in v0 -- implement TTSBackend to add one)"
    )
