import os
import tempfile
import unittest

from todocli.store import TodoStore


class TestTodoStore(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self.path)  # store should handle a missing file

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_add_and_list(self):
        store = TodoStore(self.path)
        task = store.add("write tests")
        self.assertEqual(task.id, 1)
        self.assertFalse(task.done)
        self.assertEqual(len(store.list()), 1)

    def test_mark_done(self):
        store = TodoStore(self.path)
        store.add("task one")
        self.assertTrue(store.mark_done(1))
        self.assertTrue(store.list()[0].done)
        self.assertFalse(store.mark_done(999))

    def test_remove(self):
        store = TodoStore(self.path)
        store.add("task one")
        self.assertTrue(store.remove(1))
        self.assertEqual(len(store.list()), 0)
        self.assertFalse(store.remove(1))

    def test_persistence_across_instances(self):
        store = TodoStore(self.path)
        store.add("persisted task")
        reloaded = TodoStore(self.path)
        self.assertEqual(len(reloaded.list()), 1)
        self.assertEqual(reloaded.list()[0].text, "persisted task")


if __name__ == "__main__":
    unittest.main()
