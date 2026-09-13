"""Offline tests for the three new render.py frame styles (slide, code
card, section title card): each must produce a real, correctly-sized PNG,
and the four styles (these three plus the original terminal card) must be
visually distinct from one another (not byte-identical)."""

import hashlib
import os
import tempfile
import unittest

from PIL import Image

from agent_demoforge import render


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TestNewFrameStyles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_slide_frame_is_a_real_correctly_sized_png(self):
        path = os.path.join(self.tmp, "slide.png")
        render.render_slide_frame(path, title="What agent-demoforge does", bullets=["Reads code", "Runs it", "Narrates it"])
        self.assertTrue(os.path.isfile(path))
        with Image.open(path) as img:
            self.assertEqual(img.size, (render.WIDTH, render.HEIGHT))
            self.assertEqual(img.format, "PNG")

    def test_slide_frame_without_bullets_still_renders(self):
        path = os.path.join(self.tmp, "slide2.png")
        render.render_slide_frame(path, title="Just a title, no bullets")
        self.assertTrue(os.path.isfile(path))

    def test_code_frame_is_a_real_correctly_sized_png(self):
        path = os.path.join(self.tmp, "code.png")
        render.render_code_frame(
            path, "app/store.py", 10, 14, "def add(x, y):\n    # add two things\n    return x + y\n"
        )
        self.assertTrue(os.path.isfile(path))
        with Image.open(path) as img:
            self.assertEqual(img.size, (render.WIDTH, render.HEIGHT))

    def test_code_missing_frame_renders_clean_disclosure(self):
        path = os.path.join(self.tmp, "code_missing.png")
        render.render_code_missing_frame(path, "bogus/path.py", reason="not found on disk")
        self.assertTrue(os.path.isfile(path))
        with Image.open(path) as img:
            self.assertEqual(img.size, (render.WIDTH, render.HEIGHT))

    def test_section_card_is_a_real_correctly_sized_png(self):
        path = os.path.join(self.tmp, "section.png")
        render.render_section_card(path, 2, 3, "Code Walkthrough")
        self.assertTrue(os.path.isfile(path))
        with Image.open(path) as img:
            self.assertEqual(img.size, (render.WIDTH, render.HEIGHT))

    def test_four_styles_are_visually_distinct(self):
        terminal_path = os.path.join(self.tmp, "terminal.png")
        render.render_frame(terminal_path, caption="A terminal beat", command="ls")

        slide_path = os.path.join(self.tmp, "slide3.png")
        render.render_slide_frame(slide_path, title="A slide beat", bullets=["one", "two"])

        code_path = os.path.join(self.tmp, "code3.png")
        render.render_code_frame(code_path, "x.py", 1, 3, "a = 1\nb = 2\nc = a + b\n")

        section_path = os.path.join(self.tmp, "section3.png")
        render.render_section_card(section_path, 1, 3, "Overview")

        hashes = {
            "terminal": _sha256(terminal_path),
            "slide": _sha256(slide_path),
            "code": _sha256(code_path),
            "section": _sha256(section_path),
        }
        self.assertEqual(len(set(hashes.values())), 4, f"expected 4 distinct images, got {hashes}")

        # And distinct at the pixel level too (background colors differ).
        with Image.open(terminal_path) as t, Image.open(slide_path) as s, Image.open(
            code_path
        ) as c, Image.open(section_path) as sec:
            bg_terminal = t.getpixel((5, 5))
            bg_slide = s.getpixel((5, 5))
            bg_code = c.getpixel((5, 5))
            bg_section = sec.getpixel((5, 5))
            corners = [bg_terminal, bg_slide, bg_code, bg_section]
            self.assertEqual(len(set(corners)), 4, f"expected 4 distinct corner colors, got {corners}")


if __name__ == "__main__":
    unittest.main()
