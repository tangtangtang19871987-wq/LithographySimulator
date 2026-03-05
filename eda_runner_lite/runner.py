"""LiteRunner 对外统一接口。"""

import time
from typing import Dict, Optional

from .backends import LocalLiteBackend, TaskInfo, TaskState


class LiteRunner:
    def __init__(self, log_dir: str = "~/.eda_runner_lite", env_file: Optional[str] = None):
        self.backend = LocalLiteBackend(log_dir=log_dir, env_file=env_file)

    def submit(self, cmd: str, task_id: str, workdir: Optional[str] = None) -> TaskInfo:
        return self.backend.submit(task_id=task_id, cmd=cmd, workdir=workdir)

    def status(self, task_id: str) -> TaskInfo:
        return self.backend.status(task_id)

    def result(self, task_id: str) -> Dict:
        return self.backend.result(task_id)

    def kill(self, task_id: str) -> bool:
        return self.backend.kill(task_id)

    def list_tasks(self):
        return self.backend.list_tasks()

    def wait(self, task_id: str, timeout: Optional[float] = None, interval: float = 0.2) -> TaskInfo:
        start = time.time()
        while True:
            info = self.status(task_id)
            if info.state in {TaskState.SUCCESS, TaskState.FAILED, TaskState.LOST, TaskState.UNKNOWN}:
                return info
            if timeout is not None and (time.time() - start) > timeout:
                return info
            time.sleep(interval)
