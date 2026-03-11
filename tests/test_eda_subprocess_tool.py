import time
import unittest

from eda_subprocess_tool import EdaSubprocessTool, TaskState


class EdaSubprocessToolTests(unittest.TestCase):
    def test_successful_run(self):
        tool = EdaSubprocessTool()
        result = tool.run_sync({"command": ["python", "-c", "print('ok')"]})
        self.assertEqual(result.state, TaskState.SUCCEEDED)
        self.assertEqual(result.return_code, 0)
        self.assertIn("ok", result.stdout_tail)

    def test_failed_run(self):
        tool = EdaSubprocessTool()
        result = tool.run_sync({"command": ["python", "-c", "import sys; sys.exit(7)"]})
        self.assertEqual(result.state, TaskState.FAILED)
        self.assertEqual(result.return_code, 7)

    def test_timeout(self):
        tool = EdaSubprocessTool()
        result = tool.run_sync(
            {
                "command": ["python", "-c", "import time; time.sleep(2)"],
                "timeout_seconds": 0.5,
            }
        )
        self.assertEqual(result.state, TaskState.TIMEOUT)

    def test_retry(self):
        tool = EdaSubprocessTool()
        result = tool.run_sync(
            {
                "command": ["python", "-c", "import sys; sys.exit(2)"],
                "retry_policy": {"max_retries": 1, "backoff_seconds": 0.1},
            }
        )
        self.assertEqual(result.state, TaskState.FAILED)
        self.assertEqual(result.attempt, 1)

    def test_cancel(self):
        tool = EdaSubprocessTool()
        task_id = tool.submit({"command": ["python", "-c", "import time; time.sleep(5)"]})
        time.sleep(0.4)
        tool.cancel(task_id)
        deadline = time.time() + 5
        while time.time() < deadline:
            status = tool.status(task_id)
            if status["state"] in {
                TaskState.CANCELLED.value,
                TaskState.FAILED.value,
                TaskState.TIMEOUT.value,
            }:
                break
            time.sleep(0.2)
        result = tool.get_result(task_id)
        self.assertEqual(result.state, TaskState.CANCELLED)


if __name__ == "__main__":
    unittest.main()
