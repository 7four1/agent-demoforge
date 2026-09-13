# todocli

A tiny, dependency-free command-line to-do list manager, written as a
self-contained example project.

## Install

```bash
pip install -e .
```

## Usage

```bash
# Add a task
todocli add "Write the quarterly report"

# List all tasks
todocli list

# Mark a task done
todocli done 1

# Remove a task
todocli remove 1
```

Tasks are stored in a local JSON file (`.todocli.json` in the current
directory) so the tool works immediately with no external database or
network access.

## Why this exists

This project is bundled with [agent-demoforge](https://github.com/) as a small,
fast, fully offline example repository that agent-demoforge can point itself at
to generate a narrated video demo, without depending on any external
project.
