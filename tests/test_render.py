"""Offline tests for demoforge.render: ANSI stripping and frame rendering.

None of this touches ffmpeg or the network -- render.py's core logic is
pure Pillow + string processing, so it's fully testable in isolation.
"""

import os
import tempfile
import unittest

from PIL import Image

from demoforge import render


class TestStripAnsi(unittest.TestCase):
    def test_removes_color_codes(self):
        colored = "\x1b[31mError:\x1b[0m something broke"
        self.assertEqual(render.strip_ansi(colored), "Error: something broke")

    def test_removes_cursor_movement(self):
        text = "\x1b[2K\x1b[1Ghello\x1b[Kworld"
        self.assertEqual(render.strip_ansi(text), "helloworld")

    def test_plain_text_unchanged(self):
        self.assertEqual(render.strip_ansi("plain text, no codes"), "plain text, no codes")

    def test_empty_string(self):
        self.assertEqual(render.strip_ansi(""), "")

    def test_none_safe(self):
        self.assertEqual(render.strip_ansi(None), "")


class TestWrapText(unittest.TestCase):
    def _draw(self):
        img = Image.new("RGB", (10, 10))
        from PIL import ImageDraw

        return ImageDraw.Draw(img)

    def test_short_text_single_line(self):
        draw = self._draw()
        font = render.load_font(20)
        lines = render.wrap_text("hello world", font, 1000, draw)
        self.assertEqual(lines, ["hello world"])

    def test_long_text_wraps_to_multiple_lines(self):
        draw = self._draw()
        font = render.load_font(20)
        long_text = " ".join(["word"] * 50)
        lines = render.wrap_text(long_text, font, 200, draw)
        self.assertGreater(len(lines), 1)
        # No word should be dropped in the wrap.
        self.assertEqual(" ".join(lines).split(), long_text.split())

    def test_empty_text(self):
        draw = self._draw()
        font = render.load_font(20)
        self.assertEqual(render.wrap_text("", font, 500, draw), [""])


class TestRenderFrame(unittest.TestCase):
    def test_produces_real_nonempty_png_of_expected_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "frame.png")
            result_path = render.render_frame(
                out_path,
                caption="This beat introduces the project.",
                beat_index=1,
                beat_count=3,
            )
            self.assertEqual(result_path, out_path)
            self.assertTrue(os.path.isfile(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)

            with Image.open(out_path) as img:
                self.assertEqual(img.size, (render.WIDTH, render.HEIGHT))
                self.assertEqual(img.format, "PNG")

    def test_frame_with_command_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "frame2.png")
            render.render_frame(
                out_path,
                caption="Running the tests.",
                beat_index=2,
                beat_count=3,
                command="pytest tests/",
                status="exit code 0",
                output_text="\x1b[32m5 passed\x1b[0m in 0.42s",
            )
            self.assertTrue(os.path.isfile(out_path))
            with Image.open(out_path) as img:
                self.assertEqual(img.size, (render.WIDTH, render.HEIGHT))

    def test_output_text_budget_is_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "frame3.png")
            huge_output = "line\n" * 10000
            # Should not raise or hang despite a huge output blob.
            render.render_frame(
                out_path,
                caption="Lots of output.",
                command="yes",
                output_text=huge_output,
            )
            self.assertTrue(os.path.isfile(out_path))


if __name__ == "__main__":
    unittest.main()
