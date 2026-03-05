"""后端基础类型。"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    LOST = "lost"
    UNKNOWN = "unknown"


@dataclass
class TaskInfo:
    task_id: str
    state: TaskState
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
