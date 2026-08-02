import os
import sys
import threading
import time
import unittest

from copilot_dashboard import CliSession, SessionConfig, parse_environment


class EnvironmentTests(unittest.TestCase):
    def test_parses_values_and_comments(self):
        self.assertEqual(parse_environment("# token\nFOO=bar=baz\n EMPTY=\n"), {"FOO": "bar=baz", "EMPTY": ""})

    def test_rejects_invalid_lines(self):
        with self.assertRaisesRegex(ValueError, "line 2"):
            parse_environment("OK=yes\nnot an assignment")


@unittest.skipUnless(os.name == "posix", "PTY test requires POSIX")
class SessionTests(unittest.TestCase):
    def test_streams_output_and_accepts_input(self):
        events = []
        complete = threading.Event()

        def emit(kind, data):
            events.append((kind, data))
            if kind == "status" and data.startswith("exited"):
                complete.set()

        code = "print(input().upper(), flush=True)"
        session = CliSession(SessionConfig("test", f'{sys.executable} -c "{code}"'), emit)
        session.start()
        time.sleep(0.1)
        session.send("hello\n")
        self.assertTrue(complete.wait(3))
        output = "".join(data for kind, data in events if kind == "output")
        self.assertIn("HELLO", output)
        self.assertIn(("status", "exited (0)"), events)


if __name__ == "__main__":
    unittest.main()
