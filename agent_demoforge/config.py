"""Centralized CLI-flag / environment-variable / built-in-default resolution.

Precedence for every setting, applied consistently everywhere: explicit CLI
flag > environment variable (including one loaded from a `.env` file via
python-dotenv, see `cli.py`) > built-in default.

All new/renamed environment variables use a single `AGENT_DEMOFORGE_` prefix
for consistency, with one documented exception: `ANTHROPIC_MODEL` stays as a
cross-project convention, and `AGENT_DEMOFORGE_MODEL` is accepted as a
higher-precedence alias for it (see `resolve_model` below).
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

ENV_PREFIX = "AGENT_DEMOFORGE_"

VALID_SECTIONS = ("presentation", "code_walkthrough", "live_demo")
DEFAULT_SECTIONS: List[str] = list(VALID_SECTIONS)


def resolve(cli_value, env_names: Sequence[str], default=None):
    """Return `cli_value` if given (not None/empty), else the first
    non-empty value found among `env_names` (checked in the order given --
    that order IS the precedence between multiple env var aliases), else
    `default`.
    """
    if cli_value is not None and cli_value != "":
        return cli_value
    for name in env_names:
        val = os.environ.get(name)
        if val:
            return val
    return default


def resolve_int(cli_value, env_names: Sequence[str], default: int) -> int:
    resolved = resolve(cli_value, env_names, None)
    if resolved is None:
        return default
    return int(resolved)


def resolve_model(cli_model: Optional[str], default: str) -> str:
    """Model precedence: --model flag > AGENT_DEMOFORGE_MODEL > ANTHROPIC_MODEL > default.

    AGENT_DEMOFORGE_MODEL is the project-specific env var and wins over the
    older cross-project ANTHROPIC_MODEL convention when both are set.
    """
    return resolve(cli_model, [f"{ENV_PREFIX}MODEL", "ANTHROPIC_MODEL"], default)


def parse_sections(value: Optional[str]) -> List[str]:
    """Parse a comma-separated `--sections`/env value into a validated,
    de-duplicated list of section names. Raises ValueError on an unknown
    section name. An empty/None value returns the full default order."""
    if not value:
        return list(DEFAULT_SECTIONS)
    result: List[str] = []
    for raw in value.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name not in VALID_SECTIONS:
            raise ValueError(
                f"unknown section '{name}' (valid sections: {', '.join(VALID_SECTIONS)})"
            )
        if name not in result:
            result.append(name)
    return result or list(DEFAULT_SECTIONS)


def resolve_sections(cli_value: Optional[str]) -> List[str]:
    raw = resolve(cli_value, [f"{ENV_PREFIX}SECTIONS"], None)
    return parse_sections(raw)


# (env_var_name, one-line description, default-as-shown-in-.env.example)
ENV_VAR_DOCS = [
    (
        "ANTHROPIC_API_KEY",
        "Your Anthropic API key. Required for live Explore/Script phases; "
        "without it agent-demoforge runs a disclosed offline dry-run fallback.",
        "",
    ),
    (
        "ANTHROPIC_MODEL",
        "Cross-project convention for the default Claude model. Lower "
        "precedence than AGENT_DEMOFORGE_MODEL if both are set.",
        "claude-opus-5",
    ),
    (
        f"{ENV_PREFIX}MODEL",
        "Claude model for the explore/script phases. Wins over ANTHROPIC_MODEL "
        "if both are set; --model flag wins over both.",
        "claude-opus-5",
    ),
    (
        f"{ENV_PREFIX}VOICE",
        "macOS 'say' voice name used for narration audio. Run `say -v ?` to list voices.",
        "Daniel",
    ),
    (
        f"{ENV_PREFIX}AUTHOR_NAME",
        "If set, the demo's intro beat naturally credits this person as the presenter.",
        "",
    ),
    (
        f"{ENV_PREFIX}OUT",
        "Default output directory for generated demo assets.",
        "demo_output",
    ),
    (
        f"{ENV_PREFIX}REPO",
        "Default source repo path or git URL, so `agent-demoforge generate` can be run "
        "with no positional argument.",
        "",
    ),
    (
        f"{ENV_PREFIX}SECTIONS",
        "Comma-separated subset/order of: presentation,code_walkthrough,live_demo",
        "presentation,code_walkthrough,live_demo",
    ),
    (
        f"{ENV_PREFIX}MAX_COMMANDS",
        "Hard cap on total shell commands run in the sandbox.",
        "12",
    ),
    (
        f"{ENV_PREFIX}PER_COMMAND_TIMEOUT",
        "Per-command timeout in seconds.",
        "90",
    ),
    (
        f"{ENV_PREFIX}MAX_WALL_SECONDS",
        "Hard cap on total sandbox execution wall time in seconds.",
        "480",
    ),
]


def render_env_example() -> str:
    """Render the contents of a `.env.example` file documenting every
    supported environment variable, commented out, with its purpose and
    default value."""
    lines = [
        "# agent-demoforge environment configuration",
        "#",
        "# Copy this file to .env and uncomment/edit whatever you want to override.",
        "# Precedence for every setting: CLI flag > environment variable (including one",
        "# loaded from .env) > built-in default shown below.",
        "",
    ]
    for name, doc, default in ENV_VAR_DOCS:
        lines.append(f"# {doc}")
        if default:
            lines.append(f"# {name}={default}")
        else:
            lines.append(f"# {name}=")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
