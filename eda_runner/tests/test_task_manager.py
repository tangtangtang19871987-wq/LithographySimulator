from eda_runner.runner import EDARunner
from eda_runner.task_manager import TaskManager


def test_task_manager_basic(tmp_path):
    env_file = tmp_path / "env"
    env_file.write_text("PATH=/usr/bin\n", encoding="utf-8")
    runner = EDARunner(backend="local", env_file=str(env_file), log_dir=str(tmp_path / "logs"))
    tm = TaskManager(runner, max_parallel=2, poll_interval=0.2)
    tm.add("sleep 1; echo 1", "a")
    tm.add("sleep 1; echo 2", "b")
    tm.start()
    done = tm.wait(timeout=8)
    tm.stop()
    assert done
    status = tm.status()
    assert status["completed"] + status["failed"] >= 1
