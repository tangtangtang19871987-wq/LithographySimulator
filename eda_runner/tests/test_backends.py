import time

from eda_runner.backends.base import TaskStatus
from eda_runner.backends.local import LocalBackend


def test_local_submit_and_result(tmp_path):
    env_file = tmp_path / "env"
    env_file.write_text("PATH=/usr/bin\n", encoding="utf-8")
    backend = LocalBackend(env_file=str(env_file), log_dir=str(tmp_path / "logs"))
    result = backend.submit("echo hello", "t1")
    assert result.ok
    time.sleep(0.5)
    status = backend.check_status("t1")
    assert status.status in [TaskStatus.SUCCESS, TaskStatus.RUNNING]
    output = backend.get_result("t1")
    assert "hello" in output.stdout


def test_local_kill(tmp_path):
    env_file = tmp_path / "env"
    env_file.write_text("PATH=/usr/bin\n", encoding="utf-8")
    backend = LocalBackend(env_file=str(env_file), log_dir=str(tmp_path / "logs"))
    backend.submit("sleep 10", "t2")
    time.sleep(0.2)
    assert backend.kill("t2")
