"""极简任务队列管理。"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from .backends import TaskState
from .runner import LiteRunner


@dataclass
class QueueTask:
    task_id: str
    cmd: str
    depends_on: Optional[List[str]] = None


class LiteTaskManager:
    def __init__(self, runner: LiteRunner, max_parallel: int = 2):
        self.runner = runner
        self.max_parallel = max_parallel
        self.tasks: Dict[str, QueueTask] = {}

    def add(self, task_id: str, cmd: str, depends_on: Optional[List[str]] = None) -> None:
        self.tasks[task_id] = QueueTask(task_id=task_id, cmd=cmd, depends_on=depends_on or [])

    def run_all(self) -> Dict[str, TaskState]:
        pending = set(self.tasks.keys())
        running = set()
        done: Dict[str, TaskState] = {}

        while pending or running:
            for tid in list(pending):
                if len(running) >= self.max_parallel:
                    break
                deps = self.tasks[tid].depends_on or []
                if all(done.get(d) == TaskState.SUCCESS for d in deps):
                    self.runner.submit(self.tasks[tid].cmd, tid)
                    pending.remove(tid)
                    running.add(tid)

            for tid in list(running):
                st = self.runner.status(tid).state
                if st in {TaskState.SUCCESS, TaskState.FAILED, TaskState.LOST, TaskState.UNKNOWN}:
                    running.remove(tid)
                    done[tid] = st

        return done
