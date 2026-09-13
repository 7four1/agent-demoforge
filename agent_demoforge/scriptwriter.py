"""The Script phase: one structured-output call that turns the Explore
transcript into an ordered DemoScript of DemoBeats."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class DemoBeat(BaseModel):
    narration: str
    command: Optional[str] = None
    is_setup: bool = False


class DemoScript(BaseModel):
    title: str
    beats: List[DemoBeat]


SCRIPT_INSTRUCTION = """Now produce the final demo script.

Rules:
- 5 to 9 beats total.
- `narration` is a short, natural sentence meant to be SPOKEN ALOUD by a text-to-speech \
engine: no markdown, no backticks, no asterisks, no code-formatting syntax, don't spell \
out flags or symbols character-by-character -- describe what's happening in plain spoken \
English, the way a person would narrate a screen recording.
- `command` is a real, literal shell command that can actually be run non-interactively \
against this repository to demonstrate this beat, completing within a few seconds to a \
couple of minutes -- or null for a pure narration/intro/outro beat with no command.
- Only use commands that rely on tools/files you actually confirmed exist in the repo \
while exploring.
- Mark install/setup commands (pip install, npm install, building a venv, etc.) with \
is_setup=true; these will run but their full output won't be dwelled on.
- Always include one intro beat (command=null) and one outro beat (command=null).
- Order beats so setup commands come before the beats that demonstrate functionality."""


def write_script(client, model: str, messages: list, author_name: Optional[str] = None) -> DemoScript:
    instruction = SCRIPT_INSTRUCTION
    if author_name:
        instruction += (
            f"\n- This demo is presented by {author_name}. Have the intro beat naturally "
            f"mention that {author_name} put this demo together (e.g. as part of the "
            "greeting), without sounding robotic or forced. Do not mention them again in "
            "any other beat."
        )
    response = client.messages.parse(
        model=model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        messages=messages + [{"role": "user", "content": instruction}],
        output_format=DemoScript,
    )
    return response.parsed_output
