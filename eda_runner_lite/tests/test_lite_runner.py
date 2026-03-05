from eda_runner_lite.runner import LiteRunner
from eda_runner_lite.backends.base import TaskState


def test_submit_wait_result(tmp_path):
    r = LiteRunner(log_dir=str(tmp_path / "logs"))
    r.submit("echo hello", "t1")
    st = r.wait("t1", timeout=3)
    assert st.state in {TaskState.SUCCESS, TaskState.FAILED}
    out = r.result("t1")
    assert "hello" in out["stdout"]


def test_kill(tmp_path):
    r = LiteRunner(log_dir=str(tmp_path / "logs"))
    r.submit("sleep 10", "t2")
    assert r.kill("t2")
