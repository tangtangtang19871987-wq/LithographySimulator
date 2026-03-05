from eda_runner.runner import EDARunner


def test_runner_local(tmp_path):
    env_file = tmp_path / "env"
    env_file.write_text("PATH=/usr/bin\n", encoding="utf-8")
    runner = EDARunner(backend="local", env_file=str(env_file), log_dir=str(tmp_path / "logs"))
    r = runner.submit("echo ok", "r1")
    assert r.ok
