"""EDARunner 统一接口。"""

from typing import Dict, List, Type

from .backends.base import Backend, StatusResult, SubmitResult, TaskResult
from .backends.local import LocalBackend
from .backends.lsf import LSFBackend
from .backends.ssh import SSHBackend
from .backends.socket_backend import SocketBackend


class EDARunner:
    BACKENDS: Dict[str, Type[Backend]] = {
        "local": LocalBackend,
        "lsf": LSFBackend,
        "ssh": SSHBackend,
        "socket": SocketBackend,
    }

    def __init__(self, backend: str = "local", **kwargs):
        if backend not in self.BACKENDS:
            raise ValueError(f"未知后端: {backend}，可选: {list(self.BACKENDS.keys())}")
        self._backend: Backend = self.BACKENDS[backend](**kwargs)

    @classmethod
    def register_backend(cls, name: str, backend_class: Type[Backend]):
        if not issubclass(backend_class, Backend):
            raise ValueError(f"{backend_class} 必须继承 Backend")
        cls.BACKENDS[name] = backend_class

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def submit(self, cmd: str, task_id: str, **kwargs) -> SubmitResult:
        return self._backend.submit(cmd, task_id, **kwargs)

    def status(self, task_id: str) -> StatusResult:
        return self._backend.check_status(task_id)

    def result(self, task_id: str, **kwargs) -> TaskResult:
        return self._backend.get_result(task_id, **kwargs)

    def kill(self, task_id: str) -> bool:
        return self._backend.kill(task_id)

    def cleanup(self, task_id: str) -> bool:
        return self._backend.cleanup(task_id)

    def list_tasks(self) -> List[str]:
        return self._backend.list_tasks()

    def wait(self, task_id: str, poll_interval: float = 2.0, timeout: float = None) -> StatusResult:
        import time

        start = time.time()
        while True:
            status = self.status(task_id)
            if status.is_finished:
                return status
            if timeout and (time.time() - start) > timeout:
                return status
            time.sleep(poll_interval)
