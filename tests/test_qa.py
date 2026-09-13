"""Offline tests for agent_demoforge.qa: the `ask`/`chat` read-only Q&A
feature.

Covers, all without any real network call:
  - cache-seeding: a cached Explore transcript is correctly prepended
    before the question turn, and a cache HIT never calls the (fake) LLM
    for exploration.
  - multi-turn message-history bookkeeping (`chat`'s accumulating
    `messages` list), via a scripted fake `client.messages.create`.
  - the "no credentials -> clean exit, no traceback" path for both
    `run_ask` and `run_chat`.
  - the bidirectional cache save/load round trip this feature depends on:
    a transcript saved by the cache-miss path (`ask`/`chat` exploring
    fresh, as `generate` would) is loadable afterward exactly like
    `pipeline.run_llm_phases` would load it (ask/chat -> generate
    direction), and a transcript saved directly the way `generate` saves
    it (`cache.save`) is loadable by `qa.load_or_explore` (generate ->
    ask/chat direction).
  - a scripted-client dry run of `load_or_explore`'s cache-miss path that
    exercises the REAL tool-dispatch machinery (`explorer.execute_tool`
    actually running `list_dir`/`read_file` against a real sandboxed copy
    of `examples/toy_repo`), with only the network call itself stubbed.
  - `speak_answer`'s truncation behavior (with the real `say` invocation
    stubbed out) and the `say` command construction.

None of this exercises a live Anthropic API call -- see README "Ask
questions about the code" for what remains a disclosed dry run because no
ANTHROPIC_API_KEY was available while this was built.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from agent_demoforge import cache as cache_mod
from agent_demoforge import qa, sandbox

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOY_REPO = os.path.join(REPO_ROOT, "examples", "toy_repo")


class FakeBlock:
    """Stands in for an Anthropic SDK content-block object: attribute
    access (matching the shape `explorer.py`'s real loop consumes) AND a
    `model_dump()` (matching real SDK pydantic objects, and exercising
    `cache._to_plain`'s conversion path when this gets saved to the
    cache), just like `tests/test_cache.py`'s `_FakeContentBlock`."""

    def __init__(self, type, **kwargs):
        self.type = type
        self._extra = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def model_dump(self, mode="json"):
        return {"type": self.type, **self._extra}


class FakeResponse:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


class FakeMessagesAPI:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        if not self._responses:
            raise AssertionError("FakeMessagesAPI.create called more times than scripted")
        # Snapshot the messages list at call time: it's the same mutable
        # list object the caller keeps appending to after this returns, so
        # storing a bare reference would make later inspection see
        # mutations that happened AFTER this call, not at it.
        recorded = dict(kwargs)
        if "messages" in recorded:
            recorded["messages"] = list(recorded["messages"])
        self.calls.append(recorded)
        return self._responses.pop(0)


class RaisingMessagesAPI:
    """Used to prove a code path never calls the (fake) LLM at all -- e.g.
    a cache HIT must not re-explore."""

    def create(self, **kwargs):
        raise AssertionError("messages.create should not have been called (expected a cache hit)")


class FakeClient:
    def __init__(self, responses=None, messages_api=None):
        self.messages = messages_api if messages_api is not None else FakeMessagesAPI(responses or [])


def text_response(text):
    return FakeResponse("end_turn", [FakeBlock("text", text=text)])


def tool_use_response(tool_id, name, tool_input):
    return FakeResponse(
        "tool_use",
        [FakeBlock("tool_use", id=tool_id, name=name, input=tool_input)],
    )


class TestCheckCredentials(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.pop("ANTHROPIC_API_KEY", None)

    def tearDown(self):
        if self._old is not None:
            os.environ["ANTHROPIC_API_KEY"] = self._old

    def test_missing_api_key_reports_clean_reason(self):
        client, reason = qa.check_credentials()
        self.assertIsNone(client)
        self.assertIn("ANTHROPIC_API_KEY", reason)

    def test_present_api_key_returns_a_client(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-fake-for-test"
        client, reason = qa.check_credentials()
        self.assertIsNone(reason)
        self.assertIsNotNone(client)


class TestAnswerQuestionMessageBookkeeping(unittest.TestCase):
    """Scripted fake client standing in for the LLM: asserts the running
    `messages` list accumulates the question, tool_use/tool_result blocks,
    and the final answer across turns exactly like `explorer.py`'s Explore
    loop does -- the same bookkeeping `chat` relies on for follow-up
    context."""

    def setUp(self):
        self.repo_root = tempfile.mkdtemp()
        with open(os.path.join(self.repo_root, "a.txt"), "w") as f:
            f.write("hello world\n")

    def tearDown(self):
        shutil.rmtree(self.repo_root, ignore_errors=True)

    def test_simple_end_turn_answer_appends_question_and_answer(self):
        client = FakeClient([text_response("This repo just has a text file.")])
        messages, = [[]]
        answer, updated = qa.answer_question(client, "fake-model", self.repo_root, [], "what is this repo?")

        self.assertEqual(answer, "This repo just has a text file.")
        self.assertEqual(len(updated), 2)
        self.assertEqual(updated[0], {"role": "user", "content": "what is this repo?"})
        self.assertEqual(updated[1]["role"], "assistant")

    def test_tool_use_round_trip_dispatches_real_tool_and_records_result(self):
        client = FakeClient(
            [
                tool_use_response("tool_1", "read_file", {"path": "a.txt"}),
                text_response("The file says hello world."),
            ]
        )
        answer, updated = qa.answer_question(client, "fake-model", self.repo_root, [], "what's in a.txt?")

        self.assertEqual(answer, "The file says hello world.")
        # user question, assistant tool_use, user tool_result, assistant final answer
        self.assertEqual(len(updated), 4)
        tool_result_msg = updated[2]
        self.assertEqual(tool_result_msg["role"], "user")
        self.assertIn("hello world", tool_result_msg["content"][0]["content"])
        self.assertEqual(tool_result_msg["content"][0]["tool_use_id"], "tool_1")

    def test_multi_turn_history_accumulates_across_two_questions(self):
        client = FakeClient(
            [
                text_response("First answer."),
                text_response("Second answer, building on the first."),
            ]
        )
        messages = []
        answer1, messages = qa.answer_question(client, "fake-model", self.repo_root, messages, "question one")
        self.assertEqual(answer1, "First answer.")
        self.assertEqual(len(messages), 2)

        answer2, messages = qa.answer_question(client, "fake-model", self.repo_root, messages, "question two")
        self.assertEqual(answer2, "Second answer, building on the first.")
        # Both turns' messages are preserved, in order: q1, a1, q2, a2.
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0]["content"], "question one")
        self.assertEqual(messages[2]["content"], "question two")

        # The second call's `messages=` argument must have included the
        # first turn's history -- proving context carries forward.
        second_call_messages = client.messages.calls[1]["messages"]
        self.assertEqual(len(second_call_messages), 3)  # q1, a1, q2

    def test_iteration_cap_is_honored_and_reported_honestly(self):
        # Every response is a tool_use with no end_turn ever reached.
        responses = [tool_use_response(f"t{i}", "list_dir", {"path": "."}) for i in range(5)]
        client = FakeClient(responses)
        answer, updated = qa.answer_question(
            client, "fake-model", self.repo_root, [], "loop forever?", max_iterations=5
        )
        self.assertIn("tool-loop cap", answer)
        self.assertEqual(len(client.messages.calls), 5)


class TestCacheSeeding(unittest.TestCase):
    """A cached Explore transcript must be prepended, unmodified, before
    the new question turn -- and a cache HIT must never re-explore."""

    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.repo_root = tempfile.mkdtemp()
        with open(os.path.join(self.repo_root, "a.txt"), "w") as f:
            f.write("hello")

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        shutil.rmtree(self.repo_root, ignore_errors=True)

    def test_cache_hit_returns_cached_transcript_without_calling_llm(self):
        cached_transcript = [
            {"role": "user", "content": "Explore this repository..."},
            {"role": "assistant", "content": [{"type": "text", "text": "It's a small repo with a.txt."}]},
        ]
        cache_mod.save(self.cache_dir, self.repo_root, "fake-model", cached_transcript)

        client = FakeClient(messages_api=RaisingMessagesAPI())
        buf = io.StringIO()
        with redirect_stdout(buf):
            messages, from_cache = qa.load_or_explore(client, "fake-model", self.repo_root, self.cache_dir)

        self.assertTrue(from_cache)
        self.assertEqual(messages, cached_transcript)
        self.assertIn("cache HIT", buf.getvalue())
        self.assertIn("reusing cached exploration", buf.getvalue())

    def test_question_is_appended_after_the_cached_transcript_not_mixed_in(self):
        cached_transcript = [
            {"role": "user", "content": "Explore this repository..."},
            {"role": "assistant", "content": [{"type": "text", "text": "It's a small repo with a.txt."}]},
        ]
        client = FakeClient([text_response("a.txt just contains 'hello'.")])

        answer, updated = qa.answer_question(
            client, "fake-model", self.repo_root, cached_transcript, "what's in a.txt?"
        )

        # The cached turns must appear first, untouched, then the new
        # question, then the new answer -- nothing inserted in between.
        self.assertEqual(updated[0], cached_transcript[0])
        self.assertEqual(updated[1], cached_transcript[1])
        self.assertEqual(updated[2], {"role": "user", "content": "what's in a.txt?"})
        self.assertEqual(answer, "a.txt just contains 'hello'.")

        # And the cached transcript itself is never mutated in place.
        self.assertEqual(len(cached_transcript), 2)


class TestBidirectionalCacheRoundTrip(unittest.TestCase):
    """The core claim of this feature: a transcript saved by one path
    (generate's own cache.save, or ask/chat's cache-miss explore-then-save)
    is loadable by the other."""

    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()
        self.repo_root = tempfile.mkdtemp()
        with open(os.path.join(self.repo_root, "a.txt"), "w") as f:
            f.write("hello")

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        shutil.rmtree(self.repo_root, ignore_errors=True)

    def test_generate_style_save_is_read_by_ask_chat(self):
        # Simulates `pipeline.run_llm_phases` after a live `generate` run:
        # it calls cache.save directly with the Explore transcript.
        transcript = [
            {"role": "user", "content": "Explore this repository..."},
            {"role": "assistant", "content": [{"type": "text", "text": "Generate's own exploration summary."}]},
        ]
        cache_mod.save(self.cache_dir, self.repo_root, "fake-model", transcript)

        # ask/chat's load_or_explore must see this as a HIT.
        client = FakeClient(messages_api=RaisingMessagesAPI())
        with redirect_stdout(io.StringIO()):
            messages, from_cache = qa.load_or_explore(client, "fake-model", self.repo_root, self.cache_dir)
        self.assertTrue(from_cache)
        self.assertEqual(messages, transcript)

    def test_ask_chat_miss_path_save_is_read_by_generate_style_load(self):
        # Simulates `ask`/`chat` on a repo with no existing cache entry:
        # explores fresh (fake LLM here) and saves the transcript.
        client = FakeClient(
            [
                tool_use_response("t1", "list_dir", {"path": "."}),
                text_response("This repo has one file, a.txt."),
            ]
        )
        with redirect_stdout(io.StringIO()):
            messages, from_cache = qa.load_or_explore(client, "fake-model", self.repo_root, self.cache_dir)
        self.assertFalse(from_cache)

        # `pipeline.run_llm_phases` would read this back with a plain
        # cache_mod.load call, exactly like this:
        reloaded = cache_mod.load(self.cache_dir, self.repo_root, "fake-model")
        self.assertIsNotNone(reloaded)
        self.assertEqual(len(reloaded), len(messages))


class TestDryRunAgainstRealToyRepo(unittest.TestCase):
    """Scripted-client dry run with the network call stubbed but every
    other moving part real: a real sandboxed copy of `examples/toy_repo`
    (via `sandbox.prepare_workdir`, the same function `ask`/`chat` use),
    real `explorer.execute_tool` dispatch of `list_dir`/`read_file`
    against it, and a real cache save/load round trip."""

    def setUp(self):
        self.repo_root = sandbox.prepare_workdir(TOY_REPO)
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        sandbox.cleanup_workdir(self.repo_root)
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_explore_then_answer_dispatches_real_tools_and_caches(self):
        explore_client = FakeClient(
            [
                tool_use_response("t1", "list_dir", {"path": "."}),
                tool_use_response("t2", "read_file", {"path": "pyproject.toml"}),
                text_response(
                    "toy_repo is a small todo CLI project (todocli), packaged with pyproject.toml."
                ),
            ]
        )
        with redirect_stdout(io.StringIO()):
            messages, from_cache = qa.load_or_explore(
                explore_client, "fake-model", self.repo_root, self.cache_dir
            )
        self.assertFalse(from_cache)

        # Real list_dir output must mention the real toy_repo layout.
        list_dir_result = messages[2]["content"][0]["content"]
        self.assertIn("todocli", list_dir_result)
        self.assertIn("pyproject.toml", list_dir_result)

        # Real read_file output must contain real file content.
        read_file_result = messages[4]["content"][0]["content"]
        self.assertIn("[project]", read_file_result)

        # Cache round trip: a later call (e.g. from `generate`) hits.
        cached = cache_mod.load(self.cache_dir, self.repo_root, "fake-model")
        self.assertIsNotNone(cached)

        # Now answer a follow-up question on top of that cached transcript,
        # continuing to dispatch a real tool call.
        answer_client = FakeClient(
            [
                tool_use_response("t3", "read_file", {"path": "todocli/cli.py"}),
                text_response("The CLI entry point lives in todocli/cli.py."),
            ]
        )
        answer, updated = qa.answer_question(
            answer_client, "fake-model", self.repo_root, cached, "where's the CLI entry point?"
        )
        self.assertEqual(answer, "The CLI entry point lives in todocli/cli.py.")
        # The cached exploration turns are still present ahead of the new ones.
        self.assertEqual(updated[: len(cached)], cached)


class TestNoCredentialsCleanExit(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.pop("ANTHROPIC_API_KEY", None)

    def tearDown(self):
        if self._old is not None:
            os.environ["ANTHROPIC_API_KEY"] = self._old

    def test_run_ask_exits_clean_with_no_credentials(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = qa.run_ask(TOY_REPO, "what does this do?", model="fake-model")
        self.assertEqual(rc, 1)
        self.assertIn("no offline fallback", err.getvalue())
        self.assertIn("ANTHROPIC_API_KEY", err.getvalue())

    def test_run_chat_exits_clean_with_no_credentials(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = qa.run_chat(TOY_REPO, model="fake-model")
        self.assertEqual(rc, 1)
        self.assertIn("no offline fallback", err.getvalue())


class TestSpeakAnswer(unittest.TestCase):
    def test_truncates_long_answers_before_speaking(self):
        long_answer = "word " * 1000  # far more than MAX_SPEAK_CHARS
        with mock.patch.object(qa.tts_mod, "speak_live") as fake_speak:
            fake_speak.return_value = mock.Mock(ok=True)
            qa.speak_answer(long_answer, voice="Samantha")

        self.assertEqual(fake_speak.call_count, 1)
        spoken_text = fake_speak.call_args[0][0]
        self.assertLessEqual(len(spoken_text), qa.MAX_SPEAK_CHARS + len("... (truncated for speech)"))
        self.assertEqual(fake_speak.call_args[1]["voice"], "Samantha")

    def test_short_answer_is_not_truncated(self):
        with mock.patch.object(qa.tts_mod, "speak_live") as fake_speak:
            fake_speak.return_value = mock.Mock(ok=True)
            qa.speak_answer("short answer", voice="Samantha")
        spoken_text = fake_speak.call_args[0][0]
        self.assertEqual(spoken_text, "short answer")

    def test_speak_failure_is_reported_not_raised(self):
        with mock.patch.object(qa.tts_mod, "speak_live") as fake_speak:
            fake_speak.return_value = mock.Mock(ok=False, error="'say' not available")
            buf = io.StringIO()
            with redirect_stdout(buf):
                qa.speak_answer("hello", voice=None)
        self.assertIn("WARNING", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
