"""Renders "terminal card" frame images for each demo beat using Pillow.

Pure image generation -- no ffmpeg calls live here, so this module (and
its ANSI-stripping/wrapping helpers) is fully unit-testable offline.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1280
HEIGHT = 720

BG_COLOR = (18, 18, 23)
HEADER_BG = (30, 30, 38)
CAPTION_COLOR = (240, 240, 245)
PROMPT_BG = (26, 26, 33)
PROMPT_COLOR = (120, 220, 120)
STATUS_COLOR = (230, 195, 90)
OUTPUT_BG = (10, 10, 14)
OUTPUT_COLOR = (205, 205, 215)
ACCENT_COLOR = (100, 170, 250)

OUTPUT_CHAR_BUDGET = 1400

_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

_PREFERRED_FONT_PATH = "/System/Library/Fonts/Supplemental/Andale Mono.ttf"

_font_cache = {}


def strip_ansi(text: str) -> str:
    """Remove ANSI escape/color codes from captured command output."""
    if not text:
        return ""
    return _ANSI_RE.sub("", text)


def load_font(size: int) -> ImageFont.ImageFont:
    """Load the preferred monospace font at `size`, falling back to
    Pillow's bundled default font if it isn't present on this system."""
    key = size
    if key in _font_cache:
        return _font_cache[key]
    font = None
    if os.path.exists(_PREFERRED_FONT_PATH):
        try:
            font = ImageFont.truetype(_PREFERRED_FONT_PATH, size)
        except Exception:
            font = None
    if font is None:
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:
            # Older Pillow: load_default() takes no size argument.
            font = ImageFont.load_default()
    _font_cache[key] = font
    return font


def wrap_text(text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> List[str]:
    """Greedy word-wrap `text` to fit within `max_width` pixels for `font`."""
    if text == "":
        return [""]
    words = text.split(" ")
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def render_frame(
    out_path: str,
    caption: str,
    beat_index: int = 1,
    beat_count: int = 1,
    command: Optional[str] = None,
    output_text: Optional[str] = None,
    status: Optional[str] = None,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> str:
    """Render one terminal-card frame PNG to `out_path` and return it.

    `caption` is the beat's spoken narration (shown as on-screen text too).
    `command`, if given, is rendered as a shell prompt line.
    `output_text` is the (ANSI-stripped, budget-truncated) captured command
    output so far.
    `status` is a short line like "running..." or "exit code 0".
    """
    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    title_font = load_font(28)
    body_font = load_font(20)
    mono_font = load_font(18)

    draw.rectangle([0, 0, width, 50], fill=HEADER_BG)
    draw.text(
        (20, 13),
        f"demoforge  —  beat {beat_index}/{beat_count}",
        font=body_font,
        fill=ACCENT_COLOR,
    )

    y = 72
    for line in wrap_text(caption or "", title_font, width - 40, draw):
        draw.text((20, y), line, font=title_font, fill=CAPTION_COLOR)
        y += 34
    y += 12

    if command:
        box_h = 36
        draw.rectangle([20, y, width - 20, y + box_h], fill=PROMPT_BG)
        prompt_line = f"$ {command}"
        max_chars_width = width - 60
        wrapped_cmd = wrap_text(prompt_line, mono_font, max_chars_width, draw)
        draw.text((30, y + 8), wrapped_cmd[0], font=mono_font, fill=PROMPT_COLOR)
        y += box_h + 12

    if status:
        draw.text((20, y), status, font=body_font, fill=STATUS_COLOR)
        y += 32

    if output_text is not None and y < height - 40:
        clean = strip_ansi(output_text)
        if len(clean) > OUTPUT_CHAR_BUDGET:
            clean = "...\n" + clean[-OUTPUT_CHAR_BUDGET:]
        draw.rectangle([20, y, width - 20, height - 20], fill=OUTPUT_BG)
        ty = y + 10
        line_height = 22
        for line in (clean.splitlines() or [""]):
            if ty > height - 30:
                break
            if not line.strip():
                ty += line_height
                continue
            for wrapped_line in wrap_text(line, mono_font, width - 60, draw):
                if ty > height - 30:
                    break
                draw.text((30, ty), wrapped_line, font=mono_font, fill=OUTPUT_COLOR)
                ty += line_height

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    img.save(out_path, "PNG")
    return out_path
