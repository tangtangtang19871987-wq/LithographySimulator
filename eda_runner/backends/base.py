"""Backend 基类定义。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    LOST = "lost"
    UNKNOWN = "unknown"


@dataclass
class SubmitResult:
    ok: bool
    task_id: str
    pid: Optional[int] = None
    job_id: Optional[str] = None
    confirmed: bool = False
    log_dir: Optional[str] = None
    error: Optional[str] = None
    exitcode: Optional[int] = None
    stderr: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {"ok": self.ok, "task_id": self.task_id}
        if self.pid is not None:
            result["pid"] = self.pid
        if self.job_id is not None:
            result["job_id"] = self.job_id
        if self.confirmed:
            result["confirmed"] = self.confirmed
        if self.log_dir:
            result["log_dir"] = self.log_dir
        if self.error:
            result["error"] = self.error
        if self.exitcode is not None:
            result["exitcode"] = self.exitcode
        if self.stderr:
            result["stderr"] = self.stderr
        if self.extra:
            result["extra"] = self.extra
        return result


@dataclass
class StatusResult:
    status: TaskStatus
    exitcode: Optional[int] = None
    error: Optional[str] = None
    runtime_seconds: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"status": self.status.value}
        if self.exitcode is not None:
            result["exitcode"] = self.exitcode
        if self.error:
            result["error"] = self.error
        if self.runtime_seconds is not None:
            result["runtime_seconds"] = self.runtime_seconds
        if self.raw:
            result["raw"] = self.raw
        return result

    @property
    def is_finished(self) -> bool:
        return self.status in [TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED, TaskStatus.LOST]

    @property
    def is_successful(self) -> bool:
        return self.status == TaskStatus.SUCCESS


@dataclass
class TaskResult:
    stdout: str = ""
    stderr: str = ""
    exitcode: Optional[int] = None
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"stdout": self.stdout, "stderr": self.stderr}
        if self.exitcode is not None:
            result["exitcode"] = self.exitcode
        if self.data:
            result["data"] = self.data
        return result


class Backend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def submit(self, cmd: str, task_id: str, **kwargs) -> SubmitResult:
        pass

    @abstractmethod
    def check_status(self, task_id: str) -> StatusResult:
        pass

    @abstractmethod
    def get_result(self, task_id: str, **kwargs) -> TaskResult:
        pass

    @abstractmethod
    def kill(self, task_id: str) -> bool:
        pass

    def cleanup(self, task_id: str) -> bool:
        return True

    def validate_cmd(self, cmd: str) -> bool:
        return True

    def list_tasks(self) -> List[str]:
        return []
