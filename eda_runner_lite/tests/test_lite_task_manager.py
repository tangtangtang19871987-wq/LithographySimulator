from eda_runner_lite.backends.base import TaskState
from eda_runner_lite.runner import LiteRunner
from eda_runner_lite.task_manager import LiteTaskManager


def test_run_all(tmp_path):
    r = LiteRunner(log_dir=str(tmp_path / "logs"))
    tm = LiteTaskManager(runner=r, max_parallel=2)
    tm.add("a", "echo a")
    tm.add("b", "echo b", depends_on=["a"])
    result = tm.run_all()
    assert result["a"] == TaskState.SUCCESS
    assert result["b"] == TaskState.SUCCESS
