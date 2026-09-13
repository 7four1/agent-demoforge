"""Command-line entry point: `demoforge generate <repo-path-or-git-url> --out demo_output/`."""

from __future__ import annotations

import argparse
import os
import sys

from . import pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demoforge",
        description="Point demoforge at a code repository and get back a real narrated video demo.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="generate a narrated demo video for a repository")
    gen.add_argument("source", help="local path or git URL of the repository to demo")
    gen.add_argument("--out", default="demo_output", help="output directory (default: demo_output)")
    gen.add_argument(
        "--yes",
        action="store_true",
        help="skip the interactive confirmation before executing any commands",
    )
    gen.add_argument(
        "--model",
        default=os.environ.get("ANTHROPIC_MODEL", pipeline.DEFAULT_MODEL),
        help="Claude model to use for the explore/script phases",
    )
    gen.add_argument("--max-commands", type=int, default=12, help="hard cap on total commands run")
    gen.add_argument(
        "--per-command-timeout", type=int, default=90, help="per-command timeout in seconds"
    )
    gen.add_argument(
        "--max-wall-seconds", type=int, default=480, help="hard cap on total execution wall time"
    )
    gen.add_argument(
        "--voice", default=None, help="voice name to pass to the TTS backend (macOS 'say -v')"
    )

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "generate":
        try:
            return pipeline.generate(
                source=args.source,
                out_dir=args.out,
                yes=args.yes,
                model=args.model,
                max_commands=args.max_commands,
                per_command_timeout=args.per_command_timeout,
                max_wall_seconds=args.max_wall_seconds,
                voice=args.voice,
            )
        except FileNotFoundError as e:
            print(f"demoforge: error: {e}", file=sys.stderr)
            return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
