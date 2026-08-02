import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from copilot_dashboard import CliSession, ProfileStore, SessionConfig, SessionState, parse_environment


class EnvironmentTests(unittest.TestCase):
    def test_parses_values_and_comments(self):
        self.assertEqual(parse_environment("# token\nFOO=bar=baz\n EMPTY=\n"), {"FOO": "bar=baz", "EMPTY": ""})

    def test_rejects_invalid_lines_and_names(self):
        for text in ("OK=yes\nnot an assignment", "2BAD=value"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_environment(text)


class ConfigurationTests(unittest.TestCase):
    def test_validates_command_and_directory(self):
        with self.assertRaisesRegex(ValueError, "command"):
            SessionConfig("test", "", os.getcwd()).validate()
        with self.assertRaisesRegex(ValueError, "directory"):
            SessionConfig("test", "echo ok", "/definitely/missing").validate()

    def test_profile_round_trip_and_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "profiles.json"
            store = ProfileStore(path)
            profile = SessionConfig("agent", "copilot", os.getcwd(), {"MODEL": "fast"})
            store.save([profile])
            self.assertEqual(store.load(), [profile])
            self.assertEqual(json.loads(path.read_text())[0]["name"], "agent")
            if os.name == "posix":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            path.write_text("not json")
            with self.assertRaisesRegex(ValueError, "cannot load"):
                store.load()


@unittest.skipUnless(os.name == "posix", "PTY test requires POSIX")
class SessionTests(unittest.TestCase):
    def run_session(self, code, input_text=None):
        events = []
        complete = threading.Event()

        def emit(kind, data):
            events.append((kind, data))
            if kind == "state" and data[0] == SessionState.EXITED:
                complete.set()

        session = CliSession(SessionConfig("test", f'{sys.executable} -c "{code}"'), emit)
        session.start()
        if input_text:
            time.sleep(0.1)
            session.send(input_text)
        self.assertTrue(complete.wait(3))
        return session, events

    def test_streams_output_accepts_input_and_tracks_state(self):
        session, events = self.run_session("print(input().upper(), flush=True)", "hello\n")
        output = "".join(str(data) for kind, data in events if kind == "output")
        states = [data[0] for kind, data in events if kind == "state"]
        self.assertIn("HELLO", output)
        self.assertEqual(states[:2], [SessionState.STARTING, SessionState.RUNNING])
        self.assertEqual(states[-1], SessionState.EXITED)
        self.assertFalse(session.running)

    def test_incremental_utf8_decoder_preserves_split_characters(self):
        code = "import os; b='你'.encode(); os.write(1,b[:1]); os.write(1,b[1:])"
        _, events = self.run_session(code)
        output = "".join(str(data) for kind, data in events if kind == "output")
        self.assertIn("你", output)

    def test_stop_terminates_process_group(self):
        events = []
        complete = threading.Event()
        session = CliSession(
            SessionConfig("stop", f'{sys.executable} -c "import time; time.sleep(30)"'),
            lambda kind, data: (events.append((kind, data)), complete.set() if kind == "exit" else None),
        )
        session.start()
        session.stop(grace_seconds=0.1)
        self.assertTrue(complete.wait(3))
        self.assertFalse(session.running)


if __name__ == "__main__":
    unittest.main()
