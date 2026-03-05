"""本地轻量后端。"""

import os
import signal
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .base import TaskInfo, TaskState
from ..exceptions import TaskNotFoundError
from ..utils.env import merged_env, read_env_file


class LocalLiteBackend:
    def __init__(self, log_dir: str = "~/.eda_runner_lite", env_file: Optional[str] = None):
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._env = merged_env(read_env_file(env_file)) if env_file else dict(os.environ)

    def submit(self, task_id: str, cmd: str, workdir: Optional[str] = None) -> TaskInfo:
        task_dir = self.log_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        stdout = open(task_dir / "stdout.log", "w", encoding="utf-8")
        stderr = open(task_dir / "stderr.log", "w", encoding="utf-8")
        exit_file = task_dir / "exit.code"
        cwd = str(Path(workdir).expanduser()) if workdir else os.getcwd()

        wrapper = f"{cmd}\nret=$?\necho $ret > {exit_file}\n"
        proc = subprocess.Popen(
            ["bash", "-lc", wrapper],
            cwd=cwd,
            env=self._env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )

        (task_dir / "pid").write_text(str(proc.pid), encoding="utf-8")
        return TaskInfo(task_id=task_id, state=TaskState.RUNNING, pid=proc.pid)

    def status(self, task_id: str) -> TaskInfo:
        task_dir = self.log_dir / task_id
        if not task_dir.exists():
            raise TaskNotFoundError(task_id)

        pid_file = task_dir / "pid"
        exit_file = task_dir / "exit.code"

        pid = int(pid_file.read_text(encoding="utf-8").strip()) if pid_file.exists() else None
        if exit_file.exists():
            code = int(exit_file.read_text(encoding="utf-8").strip())
            return TaskInfo(task_id=task_id, state=TaskState.SUCCESS if code == 0 else TaskState.FAILED, pid=pid, exit_code=code)

        if pid is None:
            return TaskInfo(task_id=task_id, state=TaskState.UNKNOWN, error="missing pid")

        try:
            os.kill(pid, 0)
            return TaskInfo(task_id=task_id, state=TaskState.RUNNING, pid=pid)
        except ProcessLookupError:
            return TaskInfo(task_id=task_id, state=TaskState.LOST, pid=pid, error="process vanished")

    def result(self, task_id: str) -> Dict[str, Optional[str]]:
        task_dir = self.log_dir / task_id
        if not task_dir.exists():
            raise TaskNotFoundError(task_id)

        stdout = (task_dir / "stdout.log").read_text(encoding="utf-8") if (task_dir / "stdout.log").exists() else ""
        stderr = (task_dir / "stderr.log").read_text(encoding="utf-8") if (task_dir / "stderr.log").exists() else ""
        exit_code = None
        if (task_dir / "exit.code").exists():
            exit_code = int((task_dir / "exit.code").read_text(encoding="utf-8").strip())

        return {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}

    def kill(self, task_id: str) -> bool:
        task_dir = self.log_dir / task_id
        pid_file = task_dir / "pid"
        if not pid_file.exists():
            return False
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        try:
            os.killpg(pid, signal.SIGTERM)
            return True
        except ProcessLookupError:
            return True

    def list_tasks(self) -> List[str]:
        return [p.name for p in self.log_dir.iterdir() if p.is_dir()]
