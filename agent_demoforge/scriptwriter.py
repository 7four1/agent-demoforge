"""The Script phase: one structured-output call that turns the Explore
transcript into an ordered, three-act DemoScript of DemoBeats.

The three acts (each a `section` on `DemoBeat`) are:
    presentation      -- slide-style beats: intro, pitch, key features/how-it-works.
    code_walkthrough  -- beats that point at a REAL file+line-range actually
                          read during Explore (see `CodeRef`); rendering always
                          re-reads the real file from disk, never the LLM's text.
    live_demo         -- the original mechanic: narration + optional shell
                          command + captured output.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence

from pydantic import BaseModel

Section = Literal["presentation", "code_walkthrough", "live_demo"]

SECTION_ORDER: List[str] = ["presentation", "code_walkthrough", "live_demo"]

SECTION_DISPLAY_NAMES = {
    "presentation": "Overview",
    "code_walkthrough": "Code Walkthrough",
    "live_demo": "Live Demo",
}


class CodeRef(BaseModel):
    """A reference to a range of lines in a file of the target repository.

    CRITICAL: this only ever records *which* file/lines the script-writer
    thinks are worth showing. Rendering ALWAYS re-reads the real file
    content from the sandboxed repo copy at render time using this
    path/line-range -- the actual on-screen code is never the LLM's own
    output. If the path/range doesn't exist or is out of bounds when
    rendering happens, the renderer clamps or skips gracefully rather than
    trusting this reference blindly.
    """

    path: str
    start_line: int
    end_line: int


class DemoBeat(BaseModel):
    section: Section = "live_demo"
    narration: str
    command: Optional[str] = None
    is_setup: bool = False
    bullets: Optional[List[str]] = None
    code_ref: Optional[CodeRef] = None


class DemoScript(BaseModel):
    title: str
    beats: List[DemoBeat]


def _section_rules(sections: Sequence[str]) -> str:
    parts = []
    if "presentation" in sections:
        parts.append(
            'PRESENTATION section (2-4 beats, section="presentation", command=null on all '
            "of them):\n"
            "  - One title/intro beat introducing the project by name.\n"
            '  - One "what it does" beat: a short, honest pitch drawn from the README and '
            "your own source-grounded understanding.\n"
            '  - One "key features" or "how it works" beat with `bullets` set to 3-5 short, '
            "concrete bullet points grounded in what you actually found while exploring "
            "(real module names, real capabilities) -- never generic filler like \"easy to "
            "use\" or \"fast and reliable\" unless you can point to something concrete "
            "backing it."
        )
    if "code_walkthrough" in sections:
        parts.append(
            'CODE_WALKTHROUGH section (2-4 beats, section="code_walkthrough", command=null '
            "on all of them):\n"
            "  - Each beat MUST set `code_ref` to a `path`, `start_line`, and `end_line` "
            "that refer to a REAL file and REAL line range you actually saw via a "
            "`read_file` tool call earlier in this conversation. NEVER invent a path or "
            "line range you have not actually read with `read_file` -- only reference "
            "files/lines you actually saw.\n"
            "  - `narration` explains, in plain spoken language, WHY that piece of code "
            "matters -- don't narrate literal code syntax character-by-character; the "
            "actual code text is rendered separately from real disk content, so your job "
            "is just to say why it's worth looking at."
        )
    if "live_demo" in sections:
        parts.append(
            'LIVE_DEMO section (the remaining beats, section="live_demo"):\n'
            "  - This is the hands-on part: real, literal shell commands that can actually "
            "run non-interactively against this repository, each completing within a few "
            "seconds to a couple of minutes, demonstrating genuine behavior.\n"
            "  - Only use commands that rely on tools/files you actually confirmed exist in "
            "the repo while exploring.\n"
            "  - Mark install/setup commands (pip install, npm install, building a venv, "
            "etc.) with is_setup=true; these will run but their full output won't be "
            "dwelled on.\n"
            "  - Order beats so setup commands come before the beats that demonstrate "
            "functionality."
        )
    return "\n\n".join(parts)


def build_script_instruction(sections: Sequence[str], author_name: Optional[str] = None) -> str:
    section_list = ", ".join(sections)
    instruction = f"""Now produce the final demo script.

This script has a THREE-ACT STRUCTURE, but only include beats for these requested \
sections, in this order: {section_list}. Do not produce any beat for a section not in \
that list.

General rules for every beat:
- `narration` is a short, natural sentence meant to be SPOKEN ALOUD by a text-to-speech \
engine: no markdown, no backticks, no asterisks, no code-formatting syntax, don't spell \
out flags or symbols character-by-character -- describe what's happening in plain spoken \
English, the way a person would narrate a screen recording.
- Set `section` on every beat to exactly one of: {section_list}.
- Always include one overall intro beat (command=null) at the very start of the first \
included section, and one overall outro beat (command=null) at the very end of the last \
included section.

{_section_rules(sections)}

Keep the whole script reasonably short -- roughly 5-9 beats per included section is a \
good range, fewer is fine for a small project."""
    if author_name:
        instruction += (
            f"\n\nThis demo is presented by {author_name}. Have the very first beat "
            f"naturally mention that {author_name} put this demo together (e.g. as part "
            "of the greeting), without sounding robotic or forced. Do not mention them "
            "again in any other beat."
        )
    return instruction


def write_script(
    client,
    model: str,
    messages: list,
    author_name: Optional[str] = None,
    sections: Optional[Sequence[str]] = None,
) -> DemoScript:
    sections = list(sections) if sections else list(SECTION_ORDER)
    instruction = build_script_instruction(sections, author_name=author_name)
    response = client.messages.parse(
        model=model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        messages=messages + [{"role": "user", "content": instruction}],
        output_format=DemoScript,
    )
    return response.parsed_output
