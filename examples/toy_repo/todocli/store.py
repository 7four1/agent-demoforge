"""Persistence layer for todocli.

Tasks are stored as a JSON list of objects in a file, one per project
directory. This module has no external dependencies -- it only uses the
Python standard library -- so the whole project can run with a bare
interpreter and no `pip install` of third-party packages.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import List


DEFAULT_STORE_PATH = ".todocli.json"


@dataclass
class Task:
    id: int
    text: str
    done: bool = False


class TodoStore:
    """Loads and saves a list of Task objects to a JSON file."""

    def __init__(self, path: str = DEFAULT_STORE_PATH):
        self.path = path
        self.tasks: List[Task] = self._load()

    def _load(self) -> List[Task]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [Task(**item) for item in raw]

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([asdict(t) for t in self.tasks], f, indent=2)

    def _next_id(self) -> int:
        if not self.tasks:
            return 1
        return max(t.id for t in self.tasks) + 1

    def add(self, text: str) -> Task:
        task = Task(id=self._next_id(), text=text, done=False)
        self.tasks.append(task)
        self._save()
        return task

    def list(self) -> List[Task]:
        return list(self.tasks)

    def mark_done(self, task_id: int) -> bool:
        for t in self.tasks:
            if t.id == task_id:
                t.done = True
                self._save()
                return True
        return False

    def remove(self, task_id: int) -> bool:
        before = len(self.tasks)
        self.tasks = [t for t in self.tasks if t.id != task_id]
        if len(self.tasks) != before:
            self._save()
            return True
        return False
