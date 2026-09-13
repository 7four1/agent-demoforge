"""Renders frame images for each demo beat using Pillow.

Four distinct visual styles live here, one per beat kind:
    render_frame          -- the original "terminal card" style, used for
                              live_demo beats. Unchanged.
    render_slide_frame    -- a clean, light "keynote slide" style for
                              presentation beats (title + optional bullets).
    render_code_frame     -- a dark editor-like (blue/charcoal) style for
                              code_walkthrough beats: filename + line-range
                              header, monospace code with a crude
                              keyword/string/comment highlight heuristic for
                              Python (plain monospace for everything else).
    render_section_card   -- a simple, bold, centered "Part N of M -- <Name>"
                              title card marking a section transition.

Pure image generation -- no ffmpeg calls live here, so this module (and its
ANSI-stripping/wrapping/highlighting helpers) is fully unit-testable offline.
"""

from __future__ import annotations

import keyword as _keyword_mod
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

# -- presentation "keynote slide" style: light, high-contrast, not-a-terminal --
SLIDE_BG = (250, 250, 247)
SLIDE_ACCENT = (0, 110, 200)
SLIDE_TITLE_COLOR = (20, 20, 26)
SLIDE_BODY_COLOR = (55, 58, 68)
SLIDE_BULLET_MARK_COLOR = (0, 110, 200)

# -- code walkthrough "editor" style: dark blue/charcoal, distinct from the
# near-black terminal card and from the light slide --
CODE_BG = (24, 29, 43)
CODE_HEADER_BG = (16, 20, 32)
CODE_HEADER_COLOR = (150, 190, 255)
CODE_LINE_NUM_COLOR = (90, 102, 133)
CODE_DEFAULT_COLOR = (222, 228, 240)
CODE_KEYWORD_COLOR = (255, 121, 198)
CODE_STRING_COLOR = (150, 220, 150)
CODE_COMMENT_COLOR = (105, 118, 145)
CODE_MISSING_BG = (24, 29, 43)
CODE_MISSING_COLOR = (200, 130, 130)

# -- section title card style: simple, bold, centered, distinct from all of
# the above (deep indigo, not near-black and not blue-charcoal) --
SECTION_BG = (26, 18, 46)
SECTION_ACCENT = (255, 176, 59)
SECTION_TEXT_COLOR = (240, 238, 248)

_PY_KEYWORDS = set(_keyword_mod.kwlist)
_CODE_TOKEN_RE = re.compile(
    r"(\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|#.*|[A-Za-z_][A-Za-z0-9_]*|\s+|.)"
)
_MAX_CODE_LINE_CHARS = 140

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
        f"agent-demoforge  —  beat {beat_index}/{beat_count}",
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


def render_slide_frame(
    out_path: str,
    title: str,
    bullets: Optional[List[str]] = None,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> str:
    """Render one "keynote slide" frame: a clean, light, high-contrast
    background with a title and optional bullet points -- used for
    presentation-section beats. Visually distinct from both the dark
    terminal-card and dark editor-card styles."""
    img = Image.new("RGB", (width, height), SLIDE_BG)
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, width, 14], fill=SLIDE_ACCENT)

    title_font = load_font(40)
    bullet_font = load_font(24)

    y = 110
    for line in wrap_text(title or "", title_font, width - 140, draw):
        draw.text((70, y), line, font=title_font, fill=SLIDE_TITLE_COLOR)
        y += 52

    if bullets:
        y += 30
        for bullet in bullets:
            wrapped = wrap_text(bullet, bullet_font, width - 200, draw)
            for i, line in enumerate(wrapped):
                if y > height - 50:
                    break
                if i == 0:
                    draw.ellipse([70, y + 10, 82, y + 22], fill=SLIDE_BULLET_MARK_COLOR)
                    draw.text((100, y), line, font=bullet_font, fill=SLIDE_BODY_COLOR)
                else:
                    draw.text((100, y), line, font=bullet_font, fill=SLIDE_BODY_COLOR)
                y += 36
            y += 12

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


def _highlight_python_tokens(draw: ImageDraw.ImageDraw, x: int, y: int, line: str, font) -> None:
    """Crude keyword/string/comment color heuristic for Python source
    lines -- not a real tokenizer/parser, just a regex-based approximation
    that's good enough to make code walkthrough frames visually readable."""
    if len(line) > _MAX_CODE_LINE_CHARS:
        line = line[: _MAX_CODE_LINE_CHARS - 3] + "..."
    for tok in _CODE_TOKEN_RE.findall(line):
        if not tok:
            continue
        if tok.startswith("#"):
            color = CODE_COMMENT_COLOR
        elif tok[0] in "\"'":
            color = CODE_STRING_COLOR
        elif tok in _PY_KEYWORDS:
            color = CODE_KEYWORD_COLOR
        else:
            color = CODE_DEFAULT_COLOR
        draw.text((x, y), tok, font=font, fill=color)
        x += draw.textlength(tok, font=font)


def render_code_frame(
    out_path: str,
    path: str,
    start_line: int,
    end_line: int,
    code_text: str,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> str:
    """Render one code-walkthrough frame: a filename + line-range header
    over real, monospace source text (with a crude Python syntax-highlight
    heuristic when `path` ends in `.py`; plain monospace otherwise).

    `code_text` must be REAL file content re-read from the sandboxed repo
    copy at render time by the caller (see pipeline.py) -- this function
    itself does no filesystem I/O and never trusts LLM-authored code text.
    """
    img = Image.new("RGB", (width, height), CODE_BG)
    draw = ImageDraw.Draw(img)

    header_font = load_font(22)
    mono_font = load_font(18)

    draw.rectangle([0, 0, width, 50], fill=CODE_HEADER_BG)
    header = f"{path}  (lines {start_line}-{end_line})"
    draw.text((20, 13), header, font=header_font, fill=CODE_HEADER_COLOR)

    is_python = path.endswith(".py")
    line_height = 24
    y = 70
    line_no = start_line
    lines = code_text.splitlines() or [""]
    for line in lines:
        if y > height - 30:
            break
        num_str = f"{line_no:>4}  "
        draw.text((20, y), num_str, font=mono_font, fill=CODE_LINE_NUM_COLOR)
        x = 20 + draw.textlength(num_str, font=mono_font)
        if is_python:
            _highlight_python_tokens(draw, x, y, line, mono_font)
        else:
            trimmed = line
            if len(trimmed) > _MAX_CODE_LINE_CHARS:
                trimmed = trimmed[: _MAX_CODE_LINE_CHARS - 3] + "..."
            draw.text((x, y), trimmed, font=mono_font, fill=CODE_DEFAULT_COLOR)
        y += line_height
        line_no += 1

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


def render_code_missing_frame(
    out_path: str,
    path: str,
    reason: str = "referenced file/line-range could not be found on disk",
    width: int = WIDTH,
    height: int = HEIGHT,
) -> str:
    """Fallback frame used when a code_walkthrough beat's `code_ref` could
    not be resolved to real file content (e.g. a hallucinated path) --
    clearly discloses the skip rather than ever fabricating code text."""
    img = Image.new("RGB", (width, height), CODE_MISSING_BG)
    draw = ImageDraw.Draw(img)
    header_font = load_font(22)
    body_font = load_font(24)
    draw.rectangle([0, 0, width, 50], fill=CODE_HEADER_BG)
    draw.text((20, 13), path or "(unknown path)", font=header_font, fill=CODE_HEADER_COLOR)
    y = 300
    for line in wrap_text(f"Code excerpt skipped: {reason}", body_font, width - 120, draw):
        draw.text((60, y), line, font=body_font, fill=CODE_MISSING_COLOR)
        y += 36
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


def render_section_card(
    out_path: str,
    section_index: int,
    section_count: int,
    section_name: str,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> str:
    """Render one simple, bold, centered section-transition title card,
    e.g. "Part 1 of 3" over "Overview". Visually distinct from the
    terminal-card, slide, and code-card styles (deep indigo background)."""
    img = Image.new("RGB", (width, height), SECTION_BG)
    draw = ImageDraw.Draw(img)

    label_font = load_font(28)
    title_font = load_font(52)

    label = f"PART {section_index} OF {section_count}"
    label_w = draw.textlength(label, font=label_font)
    draw.text(((width - label_w) / 2, height / 2 - 90), label, font=label_font, fill=SECTION_ACCENT)

    title_w = draw.textlength(section_name, font=title_font)
    draw.text(
        ((width - title_w) / 2, height / 2 - 20),
        section_name,
        font=title_font,
        fill=SECTION_TEXT_COLOR,
    )

    rule_w = 90
    draw.rectangle(
        [(width - rule_w) / 2, height / 2 + 60, (width + rule_w) / 2, height / 2 + 65],
        fill=SECTION_ACCENT,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    img.save(out_path, "PNG")
    return out_path
