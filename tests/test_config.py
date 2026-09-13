"""Offline tests for agent_demoforge.config: CLI-flag / env-var / default
precedence resolution, --sections parsing, model precedence, and the
.env.example renderer used by `agent-demoforge init`."""

import os
import unittest

from agent_demoforge import config


class TestResolve(unittest.TestCase):
    def setUp(self):
        for k in list(os.environ):
            if k.startswith("AGENT_DEMOFORGE_") or k == "ANTHROPIC_MODEL":
                os.environ.pop(k, None)

    def tearDown(self):
        for k in list(os.environ):
            if k.startswith("AGENT_DEMOFORGE_") or k == "ANTHROPIC_MODEL":
                os.environ.pop(k, None)

    def test_cli_value_wins_over_env_and_default(self):
        os.environ["AGENT_DEMOFORGE_VOICE"] = "Samantha"
        result = config.resolve("Daniel", ["AGENT_DEMOFORGE_VOICE"], "Alex")
        self.assertEqual(result, "Daniel")

    def test_env_value_wins_over_default_when_no_cli(self):
        os.environ["AGENT_DEMOFORGE_VOICE"] = "Samantha"
        result = config.resolve(None, ["AGENT_DEMOFORGE_VOICE"], "Alex")
        self.assertEqual(result, "Samantha")

    def test_default_used_when_neither_given(self):
        result = config.resolve(None, ["AGENT_DEMOFORGE_VOICE"], "Alex")
        self.assertEqual(result, "Alex")

    def test_first_env_name_takes_precedence_over_second(self):
        os.environ["AGENT_DEMOFORGE_MODEL"] = "claude-first"
        os.environ["ANTHROPIC_MODEL"] = "claude-second"
        result = config.resolve(None, ["AGENT_DEMOFORGE_MODEL", "ANTHROPIC_MODEL"], "default-model")
        self.assertEqual(result, "claude-first")

    def test_empty_string_cli_value_falls_through(self):
        os.environ["AGENT_DEMOFORGE_VOICE"] = "Samantha"
        result = config.resolve("", ["AGENT_DEMOFORGE_VOICE"], "Alex")
        self.assertEqual(result, "Samantha")

    def test_resolve_int_casts(self):
        os.environ["AGENT_DEMOFORGE_MAX_COMMANDS"] = "7"
        self.assertEqual(config.resolve_int(None, ["AGENT_DEMOFORGE_MAX_COMMANDS"], 12), 7)
        self.assertEqual(config.resolve_int(20, ["AGENT_DEMOFORGE_MAX_COMMANDS"], 12), 20)
        os.environ.pop("AGENT_DEMOFORGE_MAX_COMMANDS")
        self.assertEqual(config.resolve_int(None, ["AGENT_DEMOFORGE_MAX_COMMANDS"], 12), 12)


class TestResolveModel(unittest.TestCase):
    def setUp(self):
        for k in ("AGENT_DEMOFORGE_MODEL", "ANTHROPIC_MODEL"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k in ("AGENT_DEMOFORGE_MODEL", "ANTHROPIC_MODEL"):
            os.environ.pop(k, None)

    def test_cli_flag_wins_over_both_env_vars(self):
        os.environ["AGENT_DEMOFORGE_MODEL"] = "from-agent-demoforge-env"
        os.environ["ANTHROPIC_MODEL"] = "from-anthropic-env"
        self.assertEqual(config.resolve_model("from-cli", "default"), "from-cli")

    def test_agent_demoforge_model_wins_over_anthropic_model(self):
        os.environ["AGENT_DEMOFORGE_MODEL"] = "from-agent-demoforge-env"
        os.environ["ANTHROPIC_MODEL"] = "from-anthropic-env"
        self.assertEqual(config.resolve_model(None, "default"), "from-agent-demoforge-env")

    def test_falls_back_to_anthropic_model(self):
        os.environ["ANTHROPIC_MODEL"] = "from-anthropic-env"
        self.assertEqual(config.resolve_model(None, "default"), "from-anthropic-env")

    def test_falls_back_to_default(self):
        self.assertEqual(config.resolve_model(None, "default"), "default")


class TestParseSections(unittest.TestCase):
    def test_none_returns_all_three_in_canonical_order(self):
        self.assertEqual(config.parse_sections(None), list(config.VALID_SECTIONS))

    def test_empty_string_returns_default(self):
        self.assertEqual(config.parse_sections(""), list(config.VALID_SECTIONS))

    def test_single_section(self):
        self.assertEqual(config.parse_sections("live_demo"), ["live_demo"])

    def test_subset_preserves_given_order_and_dedupes(self):
        self.assertEqual(
            config.parse_sections("live_demo,live_demo,presentation"),
            ["live_demo", "presentation"],
        )

    def test_whitespace_and_case_are_tolerated(self):
        self.assertEqual(config.parse_sections(" Presentation , LIVE_DEMO "), ["presentation", "live_demo"])

    def test_unknown_section_raises(self):
        with self.assertRaises(ValueError):
            config.parse_sections("presentation,not_a_real_section")


class TestResolveSections(unittest.TestCase):
    def setUp(self):
        os.environ.pop("AGENT_DEMOFORGE_SECTIONS", None)

    def tearDown(self):
        os.environ.pop("AGENT_DEMOFORGE_SECTIONS", None)

    def test_cli_wins_over_env(self):
        os.environ["AGENT_DEMOFORGE_SECTIONS"] = "live_demo"
        self.assertEqual(
            config.resolve_sections("presentation,live_demo"), ["presentation", "live_demo"]
        )

    def test_env_used_when_no_cli(self):
        os.environ["AGENT_DEMOFORGE_SECTIONS"] = "code_walkthrough"
        self.assertEqual(config.resolve_sections(None), ["code_walkthrough"])

    def test_default_when_neither(self):
        self.assertEqual(config.resolve_sections(None), list(config.VALID_SECTIONS))


class TestRenderEnvExample(unittest.TestCase):
    def test_every_documented_var_appears_commented_out(self):
        text = config.render_env_example()
        for name, _doc, _default in config.ENV_VAR_DOCS:
            self.assertIn(f"# {name}", text)
            # Every line that sets the var must be commented out.
            for line in text.splitlines():
                if line.startswith(f"{name}="):
                    self.fail(f"{name} is set uncommented in .env.example: {line!r}")

    def test_defaults_are_shown(self):
        text = config.render_env_example()
        self.assertIn("# AGENT_DEMOFORGE_VOICE=Daniel", text)
        self.assertIn("# AGENT_DEMOFORGE_MAX_COMMANDS=12", text)

    def test_ends_with_single_trailing_newline(self):
        text = config.render_env_example()
        self.assertTrue(text.endswith("\n"))
        self.assertFalse(text.endswith("\n\n"))


if __name__ == "__main__":
    unittest.main()
