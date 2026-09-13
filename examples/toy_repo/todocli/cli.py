"""Command-line entry point for todocli.

Subcommands:
    add <text>      add a new task
    list            list all tasks, showing their id and done state
    done <id>       mark a task as done
    remove <id>     delete a task
"""

from __future__ import annotations

import argparse
import sys

from .store import TodoStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="todocli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    add_p = sub.add_parser("add", help="add a new task")
    add_p.add_argument("text", help="the task description")

    sub.add_parser("list", help="list all tasks")

    done_p = sub.add_parser("done", help="mark a task as done")
    done_p.add_argument("id", type=int, help="the task id")

    remove_p = sub.add_parser("remove", help="remove a task")
    remove_p.add_argument("id", type=int, help="the task id")

    return parser


def format_task(task) -> str:
    box = "[x]" if task.done else "[ ]"
    return f"{task.id:>3}  {box}  {task.text}"


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    store = TodoStore()

    if args.command == "add":
        task = store.add(args.text)
        print(f"Added task {task.id}: {task.text}")
    elif args.command == "list":
        tasks = store.list()
        if not tasks:
            print("No tasks yet. Add one with: todocli add \"<task text>\"")
        for task in tasks:
            print(format_task(task))
    elif args.command == "done":
        if store.mark_done(args.id):
            print(f"Marked task {args.id} as done")
        else:
            print(f"No task with id {args.id}", file=sys.stderr)
            return 1
    elif args.command == "remove":
        if store.remove(args.id):
            print(f"Removed task {args.id}")
        else:
            print(f"No task with id {args.id}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
