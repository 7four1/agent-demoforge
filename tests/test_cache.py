"""Offline tests for the best-effort Explore-phase cache
(agent_demoforge.cache): fingerprinting, and the save/load round-trip
against a synthetic transcript (including SDK-object-shaped content that
needs `_to_plain` conversion before it's JSON-serializable).

Note: this exercises the cache mechanism itself against synthetic data. It
does NOT exercise it against a real live Explore transcript, since no
ANTHROPIC_API_KEY was available in the environment this was built in --
see README "Limitations/Roadmap".
"""

import os
import tempfile
import unittest

from agent_demoforge import cache


class _FakeContentBlock:
    """Stands in for an Anthropic SDK content-block object that exposes
    `.model_dump()` but isn't itself a plain dict."""

    def __init__(self, data):
        self._data = data

    def model_dump(self, mode="json"):
        return self._data


class TestRepoFingerprint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_non_git_dir_uses_tree_fingerprint(self):
        with open(os.path.join(self.tmp, "a.txt"), "w") as f:
            f.write("hello")
        fp = cache.repo_fingerprint(self.tmp)
        self.assertTrue(fp.startswith("tree:"))

    def test_fingerprint_changes_when_file_content_size_changes(self):
        with open(os.path.join(self.tmp, "a.txt"), "w") as f:
            f.write("hello")
        fp1 = cache.repo_fingerprint(self.tmp)
        with open(os.path.join(self.tmp, "a.txt"), "w") as f:
            f.write("a much longer file content than before")
        fp2 = cache.repo_fingerprint(self.tmp)
        self.assertNotEqual(fp1, fp2)

    def test_fingerprint_stable_for_unchanged_tree(self):
        with open(os.path.join(self.tmp, "a.txt"), "w") as f:
            f.write("hello")
        self.assertEqual(cache.repo_fingerprint(self.tmp), cache.repo_fingerprint(self.tmp))

    def test_cache_key_differs_by_model(self):
        with open(os.path.join(self.tmp, "a.txt"), "w") as f:
            f.write("hello")
        key1 = cache.cache_key(self.tmp, "model-a")
        key2 = cache.cache_key(self.tmp, "model-b")
        self.assertNotEqual(key1, key2)


class TestToPlain(unittest.TestCase):
    def test_converts_nested_sdk_like_objects(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [_FakeContentBlock({"type": "text", "text": "hello there"})],
            },
        ]
        plain = cache._to_plain(messages)
        self.assertEqual(plain[0], {"role": "user", "content": "hi"})
        self.assertEqual(plain[1]["content"][0], {"type": "text", "text": "hello there"})
        # Fully JSON-serializable now.
        import json

        json.dumps(plain)


class TestSaveLoadRoundTrip(unittest.TestCase):
    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.repo_dir = tempfile.mkdtemp()
        with open(os.path.join(self.repo_dir, "a.txt"), "w") as f:
            f.write("hello")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.cache_dir, ignore_errors=True)
        shutil.rmtree(self.repo_dir, ignore_errors=True)

    def test_miss_when_nothing_cached(self):
        result = cache.load(self.cache_dir, self.repo_dir, "claude-opus-5")
        self.assertIsNone(result)

    def test_save_then_load_round_trips(self):
        messages = [{"role": "user", "content": "explore this repo"}]
        path = cache.save(self.cache_dir, self.repo_dir, "claude-opus-5", messages)
        self.assertTrue(os.path.isfile(path))
        loaded = cache.load(self.cache_dir, self.repo_dir, "claude-opus-5")
        self.assertEqual(loaded, messages)

    def test_load_miss_for_different_model(self):
        messages = [{"role": "user", "content": "explore this repo"}]
        cache.save(self.cache_dir, self.repo_dir, "claude-opus-5", messages)
        self.assertIsNone(cache.load(self.cache_dir, self.repo_dir, "claude-haiku-5"))

    def test_load_miss_after_repo_content_changes(self):
        messages = [{"role": "user", "content": "explore this repo"}]
        cache.save(self.cache_dir, self.repo_dir, "claude-opus-5", messages)
        with open(os.path.join(self.repo_dir, "a.txt"), "w") as f:
            f.write("this file changed, invalidating the fingerprint")
        self.assertIsNone(cache.load(self.cache_dir, self.repo_dir, "claude-opus-5"))

    def test_corrupt_cache_file_is_treated_as_a_miss_not_a_crash(self):
        key = cache.cache_key(self.repo_dir, "claude-opus-5")
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache.cache_path(self.cache_dir, key), "w") as f:
            f.write("{not valid json")
        self.assertIsNone(cache.load(self.cache_dir, self.repo_dir, "claude-opus-5"))


if __name__ == "__main__":
    unittest.main()
