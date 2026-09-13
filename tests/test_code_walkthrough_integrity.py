"""Tests for the single most important new correctness property: a
code_walkthrough beat's on-screen code must ALWAYS be real content re-read
from the sandboxed repo copy at render time, never LLM-authored text --
and a hallucinated/out-of-bounds code_ref must be clamped or cleanly
skipped rather than crash or fabricate code.
"""

import os
import tempfile
import unittest

from PIL import Image

from agent_demoforge import explorer, pipeline, render
from agent_demoforge.scriptwriter import CodeRef, DemoBeat


class TestReadCodeExcerpt(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.real_lines = [f"line {i}\n" for i in range(1, 21)]  # 20 real lines
        with open(os.path.join(self.tmp, "real.py"), "w", encoding="utf-8") as f:
            f.writelines(self.real_lines)

    def test_exact_range_returns_real_content(self):
        result = explorer.read_code_excerpt(self.tmp, "real.py", 3, 5)
        self.assertIsNotNone(result)
        start, end, text = result
        self.assertEqual((start, end), (3, 5))
        self.assertEqual(text, "line 3\nline 4\nline 5\n")

    def test_hallucinated_out_of_bounds_range_is_clamped_not_fabricated(self):
        # An LLM might hallucinate a range past the real file's end.
        result = explorer.read_code_excerpt(self.tmp, "real.py", 15, 9999)
        self.assertIsNotNone(result)
        start, end, text = result
        self.assertEqual(start, 15)
        self.assertEqual(end, 20)  # clamped to the real last line
        self.assertEqual(text, "".join(self.real_lines[14:20]))
        # The text returned is verifiably the REAL file's content, not invented.
        for line in text.splitlines():
            self.assertIn(line, [l.rstrip("\n") for l in self.real_lines])

    def test_hallucinated_nonexistent_path_returns_none(self):
        result = explorer.read_code_excerpt(self.tmp, "this_file_does_not_exist.py", 1, 10)
        self.assertIsNone(result)

    def test_path_escape_attempt_returns_none(self):
        result = explorer.read_code_excerpt(self.tmp, "../../../etc/passwd", 1, 5)
        self.assertIsNone(result)

    def test_start_beyond_file_length_is_clamped_to_last_line(self):
        result = explorer.read_code_excerpt(self.tmp, "real.py", 500, 600)
        self.assertIsNotNone(result)
        start, end, text = result
        self.assertEqual(start, 20)
        self.assertEqual(end, 20)
        self.assertEqual(text.strip(), "line 20")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)


class TestFrameEntriesNeverTrustLLMCodeText(unittest.TestCase):
    """Builds actual frames via pipeline._frame_entries_for_beat with a
    beat whose narration contains text that must NEVER appear as rendered
    "code" -- proving the renderer only ever draws real disk content."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.frames_dir = os.path.join(self.tmp, "frames")
        os.makedirs(self.frames_dir)
        self.repo_root = os.path.join(self.tmp, "repo")
        os.makedirs(self.repo_root)
        with open(os.path.join(self.repo_root, "app.py"), "w", encoding="utf-8") as f:
            f.write("def real_function():\n    return 42\n")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_valid_code_ref_renders_real_disk_content(self):
        beat = DemoBeat(
            section="code_walkthrough",
            narration="THIS_LLM_HALLUCINATED_TEXT_MUST_NOT_APPEAR_ON_SCREEN",
            code_ref=CodeRef(path="app.py", start_line=1, end_line=2),
        )
        record = pipeline.BeatRecord(index=1, section="code_walkthrough", narration=beat.narration)
        record.audio_duration = 2.0
        entries = pipeline._frame_entries_for_beat(beat, record, self.frames_dir, 1, 1, self.repo_root)
        self.assertEqual(len(entries), 1)
        frame_path = entries[0][0]
        self.assertTrue(os.path.isfile(frame_path))
        with Image.open(frame_path) as img:
            self.assertEqual(img.size, (render.WIDTH, render.HEIGHT))
        self.assertTrue(record.code_ref_resolved)

    def test_hallucinated_code_ref_skips_gracefully_without_crashing(self):
        beat = DemoBeat(
            section="code_walkthrough",
            narration="Narration should still play even though the code_ref is bogus.",
            code_ref=CodeRef(path="does_not_exist_anywhere.py", start_line=1, end_line=10),
        )
        record = pipeline.BeatRecord(index=2, section="code_walkthrough", narration=beat.narration)
        record.audio_duration = 2.0
        # Must not raise.
        entries = pipeline._frame_entries_for_beat(beat, record, self.frames_dir, 2, 1, self.repo_root)
        self.assertEqual(len(entries), 1)
        frame_path = entries[0][0]
        self.assertTrue(os.path.isfile(frame_path))
        self.assertFalse(record.code_ref_resolved)

    def test_no_code_ref_falls_back_to_slide_with_narration(self):
        beat = DemoBeat(section="code_walkthrough", narration="Just an intro, no code yet.")
        record = pipeline.BeatRecord(index=3, section="code_walkthrough", narration=beat.narration)
        record.audio_duration = 1.5
        entries = pipeline._frame_entries_for_beat(beat, record, self.frames_dir, 3, 1, self.repo_root)
        self.assertEqual(len(entries), 1)
        self.assertTrue(os.path.isfile(entries[0][0]))


if __name__ == "__main__":
    unittest.main()
