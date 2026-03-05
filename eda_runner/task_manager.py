"""任务管理器。"""

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import PriorityQueue
from typing import Any, Callable, Dict, List, Optional

from .backends.base import StatusResult, SubmitResult, TaskStatus
from .runner import EDARunner


@dataclass(order=True)
class Task:
    priority: int = field(compare=True)
    task_id: str = field(compare=False)
    cmd: str = field(compare=False)
    depends_on: List[str] = field(default_factory=list, compare=False)
    max_retry: int = field(default=0, compare=False)
    retry_count: int = field(default=0, compare=False)
    backend_kwargs: Dict[str, Any] = field(default_factory=dict, compare=False)
    status: TaskStatus = field(default=TaskStatus.PENDING, compare=False)
    submit_result: Optional[SubmitResult] = field(default=None, compare=False)
    start_time: Optional[datetime] = field(default=None, compare=False)
    end_time: Optional[datetime] = field(default=None, compare=False)
    error: Optional[str] = field(default=None, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "cmd": self.cmd,
            "priority": self.priority,
            "depends_on": self.depends_on,
            "max_retry": self.max_retry,
            "retry_count": self.retry_count,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error": self.error,
        }


class TaskManager:
    def __init__(self, runner: EDARunner, max_parallel: int = 10, poll_interval: float = 5.0, state_file: str = None):
        self._runner = runner
        self._max_parallel = max_parallel
        self._poll_interval = poll_interval
        self._state_file = Path(state_file).expanduser() if state_file else None
        self._tasks: Dict[str, Task] = {}
        self._queue: PriorityQueue[Task] = PriorityQueue()
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._on_task_complete: Optional[Callable[[Task], None]] = None
        self._on_task_fail: Optional[Callable[[Task], None]] = None

    def add(self, cmd: str, task_id: str, priority: int = 0, depends_on: List[str] = None, max_retry: int = 0, **backend_kwargs) -> str:
        with self._lock:
            if task_id in self._tasks:
                raise ValueError(f"任务 ID 已存在: {task_id}")
            task = Task(priority=priority, task_id=task_id, cmd=cmd, depends_on=depends_on or [], max_retry=max_retry, backend_kwargs=backend_kwargs)
            self._tasks[task_id] = task
            self._queue.put(task)
        return task_id

    def start(self, block: bool = False):
        if self._running:
            return
        self._running = True
        self._paused = False
        if block:
            self._scheduler_loop()
        else:
            self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def wait(self, task_ids: List[str] = None, timeout: float = None) -> bool:
        start = time.time()
        while True:
            with self._lock:
                tasks = [self._tasks[t] for t in task_ids if t in self._tasks] if task_ids else list(self._tasks.values())
                if all(t.status in [TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.TIMEOUT, TaskStatus.CANCELLED, TaskStatus.LOST] for t in tasks):
                    return True
            if timeout and (time.time() - start) > timeout:
                return False
            time.sleep(self._poll_interval)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "queued": sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING),
                "running": sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING),
                "completed": sum(1 for t in self._tasks.values() if t.status == TaskStatus.SUCCESS),
                "failed": sum(1 for t in self._tasks.values() if t.status in [TaskStatus.FAILED, TaskStatus.TIMEOUT]),
                "paused": self._paused,
                "tasks": {tid: t.to_dict() for tid, t in self._tasks.items()},
            }

    def results(self, task_ids: List[str] = None) -> Dict[str, Any]:
        with self._lock:
            ids = [tid for tid in task_ids if tid in self._tasks] if task_ids else list(self._tasks)
        out: Dict[str, Any] = {}
        for tid in ids:
            try:
                out[tid] = self._runner.result(tid).to_dict()
            except Exception as e:
                out[tid] = {"error": str(e)}
        return out

    def save_state(self):
        if not self._state_file:
            return
        with self._lock:
            state = {"tasks": {tid: t.to_dict() for tid, t in self._tasks.items()}, "timestamp": datetime.now().isoformat()}
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _scheduler_loop(self):
        while self._running:
            if self._paused:
                time.sleep(self._poll_interval)
                continue
            self._poll_running_tasks()
            self._submit_ready_tasks()
            self.save_state()
            time.sleep(self._poll_interval)

    def _poll_running_tasks(self):
        with self._lock:
            running = [t for t in self._tasks.values() if t.status == TaskStatus.RUNNING]
        for task in running:
            status = self._runner.status(task.task_id)
            if status.is_finished:
                with self._lock:
                    task.status = status.status
                    task.end_time = datetime.now()
                    if not status.is_successful and task.retry_count < task.max_retry:
                        task.status = TaskStatus.PENDING
                        task.retry_count += 1
                        self._queue.put(task)

    def _submit_ready_tasks(self):
        with self._lock:
            running_count = sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING)
            available = self._max_parallel - running_count
        submitted = 0
        while submitted < available:
            try:
                task = self._queue.get_nowait()
            except Exception:
                break
            if not self._check_dependencies(task):
                self._queue.put(task)
                continue
            result = self._runner.submit(task.cmd, task.task_id, **task.backend_kwargs)
            with self._lock:
                task.submit_result = result
                task.start_time = datetime.now()
                if result.ok:
                    task.status = TaskStatus.RUNNING
                else:
                    task.status = TaskStatus.FAILED
                    task.error = result.error
            submitted += 1

    def _check_dependencies(self, task: Task) -> bool:
        if not task.depends_on:
            return True
        with self._lock:
            return all(dep in self._tasks and self._tasks[dep].status == TaskStatus.SUCCESS for dep in task.depends_on)
