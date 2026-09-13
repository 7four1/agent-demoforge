"""Offline test for `agent-demoforge init`: it must actually write a
well-formed .env.example into the current directory."""

import os
import tempfile
import unittest

from agent_demoforge import cli, config


class TestCliInit(unittest.TestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp()
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self._old_cwd)
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_init_writes_env_example(self):
        rc = cli.main(["init"])
        self.assertEqual(rc, 0)
        path = os.path.join(self.tmp, ".env.example")
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for name, _doc, _default in config.ENV_VAR_DOCS:
            self.assertIn(name, text)

    def test_init_does_not_overwrite_existing_file(self):
        path = os.path.join(self.tmp, ".env.example")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# user's own content\n")
        rc = cli.main(["init"])
        self.assertEqual(rc, 1)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "# user's own content\n")


if __name__ == "__main__":
    unittest.main()
