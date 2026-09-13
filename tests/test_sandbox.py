"""Offline tests for agent-demoforge.sandbox: path confinement, copy-not-clone
isolation, and command timeout/cap enforcement. All fast and network-free."""

import os
import tempfile
import unittest

from agent_demoforge.sandbox import (
    CommandBudgetExceeded,
    PathEscapeError,
    Sandbox,
    prepare_workdir,
    safe_join,
)


class TestSafeJoin(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.base, "sub"), exist_ok=True)
        with open(os.path.join(self.base, "sub", "file.txt"), "w") as f:
            f.write("hi")

    def test_resolves_normal_relative_path(self):
        result = safe_join(self.base, "sub/file.txt")
        self.assertEqual(result, os.path.realpath(os.path.join(self.base, "sub", "file.txt")))

    def test_dot_resolves_to_base(self):
        self.assertEqual(safe_join(self.base, "."), os.path.realpath(self.base))

    def test_parent_traversal_is_rejected(self):
        with self.assertRaises(PathEscapeError):
            safe_join(self.base, "../../etc/passwd")

    def test_deep_parent_traversal_is_rejected(self):
        with self.assertRaises(PathEscapeError):
            safe_join(self.base, "sub/../../../../etc/passwd")

    def test_absolute_path_is_confined_not_escaped(self):
        # An absolute-looking path must not escape to the real filesystem root.
        result = safe_join(self.base, "/etc/passwd")
        self.assertTrue(result.startswith(os.path.realpath(self.base)))
        self.assertNotEqual(result, "/etc/passwd")


class TestPrepareWorkdirIsolation(unittest.TestCase):
    def test_local_copy_is_a_separate_directory_from_original(self):
        original = tempfile.mkdtemp()
        with open(os.path.join(original, "marker.txt"), "w") as f:
            f.write("original content")

        copy_root = prepare_workdir(original)
        try:
            self.assertNotEqual(os.path.realpath(copy_root), os.path.realpath(original))
            copied_marker = os.path.join(copy_root, "marker.txt")
            self.assertTrue(os.path.isfile(copied_marker))

            # Mutating the copy must never touch the original.
            with open(copied_marker, "w") as f:
                f.write("mutated in sandbox")
            with open(os.path.join(original, "marker.txt")) as f:
                self.assertEqual(f.read(), "original content")
        finally:
            import shutil

            shutil.rmtree(os.path.dirname(copy_root), ignore_errors=True)
            shutil.rmtree(original, ignore_errors=True)

    def test_missing_local_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            prepare_workdir("/no/such/path/should/exist/anywhere")


class TestSandboxBudgets(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp()

    def test_command_cap_enforced(self):
        sb = Sandbox(self.workdir, per_command_timeout=5, max_commands=2, max_wall_seconds=60)
        sb.run("true")
        sb.run("true")
        with self.assertRaises(CommandBudgetExceeded):
            sb.run("true")

    def test_wall_time_cap_enforced(self):
        sb = Sandbox(self.workdir, per_command_timeout=5, max_commands=100, max_wall_seconds=0)
        with self.assertRaises(CommandBudgetExceeded):
            sb.run("true")

    def test_per_command_timeout_is_respected(self):
        sb = Sandbox(self.workdir, per_command_timeout=1, max_commands=5, max_wall_seconds=60)
        result = sb.run("sleep 5")
        self.assertTrue(result.timed_out)
        self.assertIsNone(result.exit_code)
        self.assertLess(result.duration_seconds, 4)

    def test_successful_command_captures_output(self):
        sb = Sandbox(self.workdir, per_command_timeout=5, max_commands=5, max_wall_seconds=60)
        result = sb.run("echo hello-agent-demoforge")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("hello-agent-demoforge", result.stdout)
        self.assertFalse(result.timed_out)

    def test_command_runs_confined_to_workdir(self):
        sb = Sandbox(self.workdir, per_command_timeout=5, max_commands=5, max_wall_seconds=60)
        result = sb.run("pwd")
        self.assertEqual(result.stdout.strip(), os.path.realpath(self.workdir))


if __name__ == "__main__":
    unittest.main()
