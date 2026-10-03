"""Snapshot undo/redo history for the track editor."""
import unittest

from sim_ui.edit_history import EditHistory


class TestEditHistory(unittest.TestCase):
    def _snap(self, i):
        return ({"name": f"rd{i}", "points": [i]}, [{"id": i}])

    def test_push_undo_redo(self):
        h = EditHistory()
        h.clear(self._snap(0))
        h.push(self._snap(1))
        h.push(self._snap(2))

        self.assertTrue(h.can_undo)
        self.assertEqual(h.undo()[0]["name"], "rd1")
        self.assertEqual(h.undo()[0]["name"], "rd0")
        self.assertFalse(h.can_undo)
        self.assertIsNone(h.undo())

        self.assertEqual(h.redo()[0]["name"], "rd1")
        self.assertEqual(h.redo()[0]["name"], "rd2")
        self.assertFalse(h.can_redo)

    def test_push_clears_redo(self):
        h = EditHistory()
        h.clear(self._snap(0))
        h.push(self._snap(1))
        h.undo()
        h.push(self._snap(9))
        self.assertFalse(h.can_redo)
        self.assertEqual(h.undo()[0]["name"], "rd0")

    def test_limit_evicts_oldest(self):
        h = EditHistory(limit=5)
        h.clear(self._snap(0))
        for i in range(10):
            h.push(self._snap(i))
        self.assertLessEqual(h.size, 5)
        seen = []
        while True:
            s = h.undo()
            if s is None:
                break
            seen.append(s[0]["name"])
        self.assertLessEqual(len(seen), 4)

    def test_noop_push_ignored(self):
        h = EditHistory()
        h.clear(self._snap(0))
        h.push(self._snap(0))  # identical snapshot → no stack entry
        self.assertFalse(h.can_undo)
        self.assertEqual(h.size, 1)

    def test_snapshots_deep_copied(self):
        h = EditHistory()
        snap = self._snap(0)
        h.clear(snap)
        snap[0]["name"] = "mutated"
        h.push(self._snap(1))
        got = h.undo()
        self.assertEqual(got[0]["name"], "rd0")  # deep copy isolates state

    def test_clear_resets(self):
        h = EditHistory()
        h.clear(self._snap(0))
        h.push(self._snap(1))
        h.clear(self._snap(5))
        self.assertFalse(h.can_undo)
        self.assertFalse(h.can_redo)
        self.assertEqual(h.size, 1)


if __name__ == "__main__":
    unittest.main()
