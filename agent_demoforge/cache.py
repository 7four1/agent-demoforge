"""Best-effort Explore-phase transcript cache.

Keyed by a fingerprint of the (sandboxed) target repo's content plus the
model name, so re-running agent-demoforge against an unchanged repo with the
same model can skip the live Explore tool-use loop entirely and reuse its
cached transcript. The Script phase always still runs fresh -- it's one
cheap structured-output call, and `--sections`/`--author-name`/`--voice` can
change what's wanted without needing to re-explore.

Cache files live under `.agent_demoforge_cache/` in the CURRENT WORKING
DIRECTORY (never inside the sandboxed temp copy, which is deleted after
each run and would take the cache with it).

Verification note (see README "Limitations/Roadmap"): the fingerprinting
and cache read/write round-trip are unit-tested offline against synthetic
transcripts (tests/test_cache.py). This module has NOT been exercised
end-to-end against a real live Explore transcript, since no
ANTHROPIC_API_KEY was available in the environment this was built in.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Any, List, Optional

CACHE_DIR_NAME = ".agent_demoforge_cache"

_IGNORED_DIR_NAMES = {".git", "__pycache__", ".venv", "venv", "node_modules"}


def _git_head(repo_root: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    head = proc.stdout.strip()
    return head or None


def _file_tree_fingerprint(repo_root: str) -> str:
    """Hash of every file's (relative path, size), sorted -- used when
    `repo_root` isn't a git repo (or `git` isn't available)."""
    entries = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(d for d in dirnames if d not in _IGNORED_DIR_NAMES)
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, repo_root)
            try:
                size = os.path.getsize(full)
            except OSError:
                size = -1
            entries.append(f"{rel}:{size}")
    joined = "\n".join(sorted(entries))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def repo_fingerprint(repo_root: str) -> str:
    """A content fingerprint for `repo_root`: its git HEAD commit if it's a
    git repo, else a hash of its full relative-path+size file tree."""
    head = _git_head(repo_root)
    if head:
        return f"git:{head}"
    return f"tree:{_file_tree_fingerprint(repo_root)}"


def cache_key(repo_root: str, model: str) -> str:
    fp = repo_fingerprint(repo_root)
    return hashlib.sha256(f"{fp}|{model}".encode("utf-8")).hexdigest()[:32]


def _to_plain(obj: Any) -> Any:
    """Recursively convert Anthropic SDK response objects (pydantic models)
    into plain JSON-serializable dicts/lists, leaving already-plain
    dicts/lists/primitives untouched. This is what makes a cached
    transcript safely resendable as `messages=` on a later API call."""
    if hasattr(obj, "model_dump"):
        return _to_plain(obj.model_dump(mode="json"))
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"{key}.json")


def load(cache_dir: str, repo_root: str, model: str) -> Optional[List[dict]]:
    """Return the cached Explore transcript for (repo_root, model), or None
    on a cache miss -- including any read/parse failure. A broken/corrupt
    cache entry must never crash the pipeline, only be treated as a miss."""
    key = cache_key(repo_root, model)
    path = cache_path(cache_dir, key)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        messages = data["messages"]
        if not isinstance(messages, list):
            return None
        return messages
    except Exception:
        return None


def save(cache_dir: str, repo_root: str, model: str, messages: List[Any]) -> str:
    """Persist `messages` (the Explore transcript) to the cache. Best
    effort: on any failure, this raises to the caller, which should treat a
    failed save as non-fatal (log and continue)."""
    os.makedirs(cache_dir, exist_ok=True)
    key = cache_key(repo_root, model)
    path = cache_path(cache_dir, key)
    payload = {"model": model, "messages": _to_plain(messages)}
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)
    return path
