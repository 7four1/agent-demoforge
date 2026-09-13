"""Offline tests for --sections filtering: the live Script-writing
instruction builder (scriptwriter.build_script_instruction) and the offline
fallback script builder (pipeline.build_fallback_script) must both respect
the requested section subset/order and produce beats only for it."""

import os
import shutil
import tempfile
import unittest

from agent_demoforge import pipeline, scriptwriter


class TestBuildScriptInstruction(unittest.TestCase):
    def test_all_sections_mentions_all_three_rule_blocks(self):
        instruction = scriptwriter.build_script_instruction(scriptwriter.SECTION_ORDER)
        self.assertIn("PRESENTATION section", instruction)
        self.assertIn("CODE_WALKTHROUGH section", instruction)
        self.assertIn("LIVE_DEMO section", instruction)

    def test_excluding_code_walkthrough_omits_its_rules(self):
        instruction = scriptwriter.build_script_instruction(["presentation", "live_demo"])
        self.assertIn("PRESENTATION section", instruction)
        self.assertNotIn("CODE_WALKTHROUGH section", instruction)
        self.assertIn("LIVE_DEMO section", instruction)

    def test_live_demo_only_matches_original_behavior_scope(self):
        instruction = scriptwriter.build_script_instruction(["live_demo"])
        self.assertNotIn("PRESENTATION section", instruction)
        self.assertNotIn("CODE_WALKTHROUGH section", instruction)
        self.assertIn("LIVE_DEMO section", instruction)

    def test_author_name_mentioned_when_given(self):
        instruction = scriptwriter.build_script_instruction(["live_demo"], author_name="Ada Lovelace")
        self.assertIn("Ada Lovelace", instruction)

    def test_author_name_omitted_when_not_given(self):
        instruction = scriptwriter.build_script_instruction(["live_demo"])
        self.assertNotIn("presented by", instruction)


class TestBuildFallbackScriptSections(unittest.TestCase):
    """Uses the real bundled examples/toy_repo as the target repo, since
    build_fallback_script reads real files from disk."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "toy_repo"
        )
        assert os.path.isdir(cls.repo_root), "examples/toy_repo must exist for this test"

    def test_default_sections_produce_all_three(self):
        script = pipeline.build_fallback_script(self.repo_root, source=self.repo_root)
        sections_seen = {b.section for b in script.beats}
        self.assertEqual(sections_seen, {"presentation", "code_walkthrough", "live_demo"})

    def test_live_demo_only_produces_no_other_sections(self):
        script = pipeline.build_fallback_script(
            self.repo_root, source=self.repo_root, sections=["live_demo"]
        )
        sections_seen = {b.section for b in script.beats}
        self.assertEqual(sections_seen, {"live_demo"})
        # Real commands should still reference the real console script.
        commands = [b.command for b in script.beats if b.command]
        self.assertTrue(any("todocli" in c for c in commands))

    def test_presentation_and_live_demo_skips_code_walkthrough(self):
        script = pipeline.build_fallback_script(
            self.repo_root, source=self.repo_root, sections=["presentation", "live_demo"]
        )
        sections_seen = {b.section for b in script.beats}
        self.assertEqual(sections_seen, {"presentation", "live_demo"})

    def test_code_walkthrough_beats_reference_real_files_and_lines(self):
        script = pipeline.build_fallback_script(
            self.repo_root, source=self.repo_root, sections=["code_walkthrough"]
        )
        cw_beats = [b for b in script.beats if b.section == "code_walkthrough" and b.code_ref]
        self.assertGreater(len(cw_beats), 0)
        for beat in cw_beats:
            full_path = os.path.join(self.repo_root, beat.code_ref.path)
            self.assertTrue(os.path.isfile(full_path), f"{beat.code_ref.path} must be a real file")
            with open(full_path, encoding="utf-8") as f:
                total_lines = len(f.readlines())
            self.assertGreaterEqual(beat.code_ref.start_line, 1)
            self.assertLessEqual(beat.code_ref.end_line, total_lines)
            self.assertLessEqual(beat.code_ref.start_line, beat.code_ref.end_line)

    def test_presentation_bullets_are_grounded_not_placeholder(self):
        script = pipeline.build_fallback_script(
            self.repo_root, source=self.repo_root, sections=["presentation"]
        )
        bullet_beats = [b for b in script.beats if b.bullets]
        self.assertGreater(len(bullet_beats), 0)
        bullets = bullet_beats[0].bullets
        joined = " ".join(bullets).lower()
        # Real, concrete facts about toy_repo -- not generic filler.
        self.assertTrue("todocli" in joined or "python source file" in joined)

    def test_single_section_list_has_intro_and_outro_in_that_section(self):
        script = pipeline.build_fallback_script(
            self.repo_root, source=self.repo_root, sections=["presentation"]
        )
        self.assertTrue(all(b.section == "presentation" for b in script.beats))
        self.assertGreaterEqual(len(script.beats), 2)


if __name__ == "__main__":
    unittest.main()
