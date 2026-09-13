"""Command-line entry point: `agent-demoforge generate <repo-path-or-git-url> --out demo_output/`.

Loads a `.env` file from the current working directory (if present) before
parsing arguments, so every setting documented in `config.ENV_VAR_DOCS` can
be set via `.env` as well as its CLI flag. Precedence everywhere: explicit
CLI flag > environment variable (`.env` included) > built-in default.
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import find_dotenv, load_dotenv

from . import config, pipeline, qa


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-demoforge",
        description="Point agent-demoforge at a code repository and get back a real narrated video demo.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="generate a narrated demo video for a repository")
    gen.add_argument(
        "source",
        nargs="?",
        default=None,
        help="local path or git URL of the repository to demo. Optional if "
        f"{config.ENV_PREFIX}REPO is set (directly or via .env).",
    )
    gen.add_argument(
        "--out",
        default=None,
        help=f"output directory (default: demo_output, env: {config.ENV_PREFIX}OUT)",
    )
    gen.add_argument(
        "--yes",
        action="store_true",
        help="skip the interactive confirmation before executing any commands",
    )
    gen.add_argument(
        "--model",
        default=None,
        help="Claude model to use for the explore/script phases. Precedence: --model > "
        f"{config.ENV_PREFIX}MODEL > ANTHROPIC_MODEL > built-in default "
        f"({pipeline.DEFAULT_MODEL}).",
    )
    gen.add_argument(
        "--max-commands",
        type=int,
        default=None,
        help=f"hard cap on total commands run (default: 12, env: {config.ENV_PREFIX}MAX_COMMANDS)",
    )
    gen.add_argument(
        "--per-command-timeout",
        type=int,
        default=None,
        help="per-command timeout in seconds (default: 90, env: "
        f"{config.ENV_PREFIX}PER_COMMAND_TIMEOUT)",
    )
    gen.add_argument(
        "--max-wall-seconds",
        type=int,
        default=None,
        help="hard cap on total execution wall time (default: 480, env: "
        f"{config.ENV_PREFIX}MAX_WALL_SECONDS)",
    )
    gen.add_argument(
        "--voice",
        default=None,
        help="voice name to pass to the TTS backend (macOS 'say -v'). Defaults to "
        f"'Samantha', a calm, soft-spoken female voice (env: {config.ENV_PREFIX}VOICE). "
        "Run `say -v ?` to list every voice installed on this Mac.",
    )
    gen.add_argument(
        "--author-name",
        default=None,
        help="if set, the demo's intro beat will naturally mention this person as the "
        "one who put the demo together (e.g. --author-name 'Srinivasarao Polagani'). "
        f"Also configurable via the {config.ENV_PREFIX}AUTHOR_NAME env var. Omit for no mention.",
    )
    gen.add_argument(
        "--sections",
        default=None,
        help="comma-separated subset of presentation,code_walkthrough,live_demo to "
        f"include (default: all three). Also configurable via {config.ENV_PREFIX}SECTIONS.",
    )
    gen.add_argument(
        "--no-cache",
        action="store_true",
        help="force a fresh Explore phase even if a cached transcript exists for this "
        "repo content + model (see .agent_demoforge_cache/).",
    )

    def _add_qa_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--model",
            default=None,
            help="Claude model to use. Precedence: --model > "
            f"{config.ENV_PREFIX}MODEL > ANTHROPIC_MODEL > built-in default "
            f"({pipeline.DEFAULT_MODEL}).",
        )
        p.add_argument(
            "--voice",
            default=None,
            help="voice name to pass to 'say -v' when --speak is given. Defaults to "
            f"'Samantha' (env: {config.ENV_PREFIX}VOICE).",
        )
        p.add_argument(
            "--speak",
            action="store_true",
            help="also speak each answer aloud live through the speakers via macOS 'say' "
            "at the resolved --voice (macOS-only; off by default).",
        )

    ask = sub.add_parser(
        "ask",
        help="ask a one-shot, read-only question about a repository, grounded in its code",
    )
    ask.add_argument(
        "args",
        nargs="+",
        metavar="[<repo-path-or-git-url>] <question>",
        help="the question to ask, optionally preceded by a repo path/URL. If the repo is "
        f"omitted, {config.ENV_PREFIX}REPO must be set (directly or via .env).",
    )
    _add_qa_common_args(ask)

    chat = sub.add_parser(
        "chat",
        help="start an interactive, read-only multi-turn Q&A session about a repository",
    )
    chat.add_argument(
        "source",
        nargs="?",
        default=None,
        help="local path or git URL of the repository to chat about. Optional if "
        f"{config.ENV_PREFIX}REPO is set (directly or via .env).",
    )
    _add_qa_common_args(chat)

    sub.add_parser(
        "init",
        help="write a .env.example file documenting every AGENT_DEMOFORGE_* setting",
    )

    return parser


def _run_init() -> int:
    target = ".env.example"
    if os.path.exists(target):
        print(f"agent-demoforge: '{target}' already exists; not overwriting it.")
        print("agent-demoforge: delete it first (or edit it directly) if you want a fresh copy.")
        return 1
    with open(target, "w", encoding="utf-8") as f:
        f.write(config.render_env_example())
    print(f"agent-demoforge: wrote {os.path.abspath(target)}")
    print("agent-demoforge: copy it to .env and edit the values you want to override:")
    print(f"    cp {target} .env")
    return 0


def main(argv=None) -> int:
    # find_dotenv(usecwd=True) is required here: load_dotenv()'s own default
    # search walks up from the *calling code's file location* (i.e. wherever
    # agent_demoforge is installed), not the user's actual current working
    # directory. Since the whole point is "load .env from the directory the
    # user is running `agent-demoforge` from", we resolve the path ourselves
    # relative to the CWD and hand it to load_dotenv() explicitly.
    load_dotenv(find_dotenv(usecwd=True))

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "init":
        return _run_init()

    if args.cmd == "generate":
        try:
            sections = config.resolve_sections(args.sections)
        except ValueError as e:
            print(f"agent-demoforge: error: {e}", file=sys.stderr)
            return 1

        model = config.resolve_model(args.model, pipeline.DEFAULT_MODEL)
        source = config.resolve(args.source, [f"{config.ENV_PREFIX}REPO"], None)
        out_dir = config.resolve(args.out, [f"{config.ENV_PREFIX}OUT"], "demo_output")
        voice = config.resolve(args.voice, [f"{config.ENV_PREFIX}VOICE"], "Samantha")
        author_name = config.resolve(args.author_name, [f"{config.ENV_PREFIX}AUTHOR_NAME"], None)
        max_commands = config.resolve_int(args.max_commands, [f"{config.ENV_PREFIX}MAX_COMMANDS"], 12)
        per_command_timeout = config.resolve_int(
            args.per_command_timeout, [f"{config.ENV_PREFIX}PER_COMMAND_TIMEOUT"], 90
        )
        max_wall_seconds = config.resolve_int(
            args.max_wall_seconds, [f"{config.ENV_PREFIX}MAX_WALL_SECONDS"], 480
        )

        if not source:
            print(
                f"agent-demoforge: error: no repository given -- pass a positional "
                f"<repo-path-or-git-url>, or set {config.ENV_PREFIX}REPO (directly or via .env)",
                file=sys.stderr,
            )
            return 1

        try:
            return pipeline.generate(
                source=source,
                out_dir=out_dir,
                yes=args.yes,
                model=model,
                max_commands=max_commands,
                per_command_timeout=per_command_timeout,
                max_wall_seconds=max_wall_seconds,
                voice=voice,
                author_name=author_name,
                sections=sections,
                no_cache=args.no_cache,
            )
        except FileNotFoundError as e:
            print(f"agent-demoforge: error: {e}", file=sys.stderr)
            return 1

    if args.cmd in ("ask", "chat"):
        model = config.resolve_model(args.model, pipeline.DEFAULT_MODEL)
        voice = config.resolve(args.voice, [f"{config.ENV_PREFIX}VOICE"], "Samantha")

        if args.cmd == "ask":
            # args.args is 1+ tokens: either just the question (repo falls
            # back to AGENT_DEMOFORGE_REPO), or "<repo> <question...>".
            # Same fallback logic as `generate`'s positional source arg
            # (config.resolve), just applied after we've worked out which
            # token(s) are the repo vs. the question.
            env_repo = config.resolve(None, [f"{config.ENV_PREFIX}REPO"], None)
            if len(args.args) == 1:
                source, question = env_repo, args.args[0]
            else:
                source, question = args.args[0], " ".join(args.args[1:])

            if not source:
                print(
                    f"agent-demoforge: error: no repository given -- pass "
                    f"<repo-path-or-git-url> before the question, or set "
                    f"{config.ENV_PREFIX}REPO (directly or via .env)",
                    file=sys.stderr,
                )
                return 1
            if not question or not question.strip():
                print("agent-demoforge: error: no question given", file=sys.stderr)
                return 1

            return qa.run_ask(source, question, model=model, voice=voice, speak=args.speak)

        source = config.resolve(args.source, [f"{config.ENV_PREFIX}REPO"], None)
        if not source:
            print(
                f"agent-demoforge: error: no repository given -- pass a positional "
                f"<repo-path-or-git-url>, or set {config.ENV_PREFIX}REPO (directly or via .env)",
                file=sys.stderr,
            )
            return 1
        return qa.run_chat(source, model=model, voice=voice, speak=args.speak)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
