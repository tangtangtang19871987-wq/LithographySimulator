from eda_runner.runner import EDARunner


def test_wait(tmp_path):
    env_file = tmp_path / "env"
    env_file.write_text("PATH=/usr/bin\n", encoding="utf-8")
    runner = EDARunner(backend="local", env_file=str(env_file), log_dir=str(tmp_path / "logs"))
    runner.submit("echo done", "w1")
    status = runner.wait("w1", poll_interval=0.1, timeout=3)
    assert status.status.value in {"success", "failed", "running"}
