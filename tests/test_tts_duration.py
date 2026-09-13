"""Offline test for AIFF duration reading.

If the macOS `say` command is available in this test environment, this
synthesizes a real short AIFF file and checks the duration read back from
its COMM chunk is a sane, positive number close to what's expected for the
text's length. If `say` isn't available (non-macOS CI), this test skips
cleanly instead of failing.
"""

import shutil
import subprocess
import tempfile
import unittest
import os

from agent_demoforge.tts import MacSayTTSBackend, read_aiff_duration


@unittest.skipUnless(shutil.which("say"), "macOS 'say' command not available in this environment")
class TestAiffDuration(unittest.TestCase):
    def test_real_say_output_has_sane_duration(self):
        backend = MacSayTTSBackend()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "test.aiff")
            text = "This is a short test sentence for agent-demoforge."
            result = backend.synthesize(text, out_path)

            self.assertTrue(result.ok, msg=result.error)
            self.assertTrue(os.path.isfile(out_path))
            self.assertGreater(os.path.getsize(out_path), 44)  # more than a bare header

            duration = read_aiff_duration(out_path)
            # Sanity bounds: a short sentence spoken by `say` should land
            # comfortably between half a second and 15 seconds.
            self.assertGreater(duration, 0.5)
            self.assertLess(duration, 15.0)
            self.assertAlmostEqual(duration, result.duration_seconds, places=5)

    def test_nonexistent_file_raises(self):
        with self.assertRaises(Exception):
            read_aiff_duration("/no/such/file.aiff")

    def test_not_an_aiff_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            bogus = os.path.join(tmp, "not_audio.aiff")
            with open(bogus, "wb") as f:
                f.write(b"not an aiff file at all, just bytes")
            with self.assertRaises(ValueError):
                read_aiff_duration(bogus)


if __name__ == "__main__":
    unittest.main()
