"""Sandboxing: safe temp-copy/clone of a target repo, path confinement, and
capped shell command execution.

Safety is the whole point of this module: agent-demoforge must never run anything
against the user's original repository. Every entry point here either
(a) creates a brand-new temp directory copy/clone, or (b) operates strictly
inside a directory that was already confirmed to be such a copy.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Optional


GIT_URL_PREFIXES = ("http://", "https://", "git@", "ssh://", "git://")

SETUP_FILES = ("pyproject.toml", "setup.py", "requirements.txt")

IGNORED_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".agent_demoforge_venv",
}


class PathEscapeError(Exception):
    """Raised when a requested path would resolve outside the sandbox root."""


def is_git_url(source: str) -> bool:
    return source.startswith(GIT_URL_PREFIXES) or source.rstrip("/").endswith(".git")


def safe_join(base_dir: str, user_path: str) -> str:
    """Resolve `user_path` relative to `base_dir`, guaranteeing the result
    stays inside `base_dir`.

    Absolute-looking input paths are treated as relative to `base_dir`
    (their leading slash is stripped) rather than escaping to the real
    filesystem root, and any `..` traversal that would leave `base_dir` is
    rejected with PathEscapeError.
    """
    real_base = os.path.realpath(base_dir)
    candidate_raw = (user_path or ".").strip() or "."
    normalized = os.path.normpath(candidate_raw)
    if os.path.isabs(normalized):
        normalized = normalized.lstrip(os.sep)
        if not normalized:
            normalized = "."
    joined = os.path.join(real_base, normalized)
    resolved = os.path.realpath(joined)
    if resolved != real_base and not resolved.startswith(real_base + os.sep):
        raise PathEscapeError(
            f"path '{user_path}' resolves outside the sandboxed repository root"
        )
    return resolved


def prepare_workdir(source: str) -> str:
    """Create a fresh temp directory containing an isolated copy (local
    path) or shallow clone (git URL) of `source`. Returns the path to the
    copy/clone -- never the original path.
    """
    tmp_parent = tempfile.mkdtemp(prefix="agent_demoforge_")
    target = os.path.join(tmp_parent, "repo")

    if is_git_url(source):
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", source, target],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.CalledProcessError as e:
            shutil.rmtree(tmp_parent, ignore_errors=True)
            raise RuntimeError(f"git clone failed: {e.stderr}") from e
        except subprocess.TimeoutExpired as e:
            shutil.rmtree(tmp_parent, ignore_errors=True)
            raise RuntimeError("git clone timed out after 300s") from e
        return target

    source_abs = os.path.realpath(os.path.expanduser(source))
    if not os.path.isdir(source_abs):
        shutil.rmtree(tmp_parent, ignore_errors=True)
        raise FileNotFoundError(f"local repo path does not exist: {source}")

    shutil.copytree(
        source_abs,
        target,
        ignore=shutil.ignore_patterns(*IGNORED_DIR_NAMES, "*.pyc"),
    )
    return target


def cleanup_workdir(repo_root: str) -> None:
    """Remove the temp parent directory containing `repo_root` (best effort)."""
    parent = os.path.dirname(os.path.normpath(repo_root))
    shutil.rmtree(parent, ignore_errors=True)


@dataclass
class CommandResult:
    command: str
    exit_code: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float


class CommandBudgetExceeded(Exception):
    """Raised when a sandbox session would exceed its command-count or
    wall-clock budget."""


class Sandbox:
    """Executes shell commands confined to `workdir`, with a per-command
    timeout, a hard cap on total commands, and a hard cap on total wall
    time across the whole session.
    """

    def __init__(
        self,
        workdir: str,
        per_command_timeout: int = 90,
        max_commands: int = 12,
        max_wall_seconds: int = 480,
    ):
        self.workdir = workdir
        self.per_command_timeout = per_command_timeout
        self.max_commands = max_commands
        self.max_wall_seconds = max_wall_seconds
        self.venv_python: Optional[str] = None
        self._commands_run = 0
        self._start_time = time.monotonic()

    def is_python_project(self) -> bool:
        return any(os.path.isfile(os.path.join(self.workdir, f)) for f in SETUP_FILES)

    def setup_venv(self, timeout: int = 180) -> str:
        """Create a fresh virtualenv inside the sandboxed copy and return the
        path to its python interpreter."""
        venv_dir = os.path.join(self.workdir, ".agent_demoforge_venv")
        subprocess.run(
            [sys.executable, "-m", "venv", venv_dir],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        bin_dir = "Scripts" if os.name == "nt" else "bin"
        python_name = "python.exe" if os.name == "nt" else "python"
        self.venv_python = os.path.join(venv_dir, bin_dir, python_name)
        return self.venv_python

    def _check_budget(self) -> None:
        if self._commands_run >= self.max_commands:
            raise CommandBudgetExceeded(
                f"command cap of {self.max_commands} commands reached"
            )
        elapsed = time.monotonic() - self._start_time
        if elapsed >= self.max_wall_seconds:
            raise CommandBudgetExceeded(
                f"wall-time cap of {self.max_wall_seconds}s reached"
                f" ({elapsed:.0f}s elapsed)"
            )

    def run(self, command: str) -> CommandResult:
        """Run `command` via the shell, cwd-confined to self.workdir."""
        self._check_budget()
        self._commands_run += 1

        env = os.environ.copy()
        if self.venv_python:
            bin_dir = os.path.dirname(self.venv_python)
            env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
            env["VIRTUAL_ENV"] = os.path.dirname(bin_dir)

        start = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.per_command_timeout,
            )
            exit_code: Optional[int] = proc.returncode
            stdout, stderr = proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            timed_out = True
            exit_code = None

            def _decode(x):
                if x is None:
                    return ""
                return x if isinstance(x, str) else x.decode("utf-8", errors="replace")

            stdout, stderr = _decode(e.stdout), _decode(e.stderr)
        duration = time.monotonic() - start

        return CommandResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_seconds=duration,
        )
