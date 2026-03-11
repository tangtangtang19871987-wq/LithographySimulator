"""Subprocess-based EDA tool runner for agent workflows.

Design goals:
- Minimal dependencies: Python standard library only.
- Support local execution and pluggable scheduler backends.
- Handle login-shell environment differences, optional preflight checks,
  retries, timeout, graceful/forceful termination, and resource cleanup.
- Offer sync/async APIs suitable for LLM-agent tool invocation.
"""

from __future__ import annotations

import enum
import json
import os
import queue
import shlex
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple


class TaskState(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class RetryPolicy:
    max_retries: int = 0
    backoff_seconds: float = 2.0


@dataclass
class TaskRequest:
    command: List[str]
    cwd: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    login_shell: bool = False
    shell_init_script: Optional[str] = None
    timeout_seconds: Optional[float] = None
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    preflight_checks: List[List[str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    state: TaskState
    return_code: Optional[int]
    started_at: Optional[float]
    ended_at: Optional[float]
    attempt: int
    stdout_tail: str
    stderr_tail: str
    error_message: Optional[str] = None
    backend_job_id: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at is None or self.ended_at is None:
            return None
        return self.ended_at - self.started_at


@dataclass
class RunningTask:
    task_id: str
    request: TaskRequest
    state: TaskState = TaskState.PENDING
    attempt: int = 0
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    return_code: Optional[int] = None
    backend_job_id: Optional[str] = None
    stdout_buffer: str = ""
    stderr_buffer: str = ""
    error_message: Optional[str] = None
    cancel_requested: bool = False


class BackendHandle(Protocol):
    job_id: Optional[str]


class ExecutionBackend(Protocol):
    def submit(self, request: TaskRequest) -> BackendHandle:
        ...

    def poll(self, handle: BackendHandle) -> Tuple[bool, Optional[int]]:
        """Return (finished, return_code). return_code can be None if unknown yet."""

    def read_incremental_output(self, handle: BackendHandle) -> Tuple[str, str]:
        ...

    def cancel(self, handle: BackendHandle) -> None:
        ...

    def cleanup(self, handle: BackendHandle) -> None:
        ...


@dataclass
class LocalHandle:
    process: subprocess.Popen
    job_id: Optional[str] = None


class LocalSubprocessBackend:
    """Local subprocess backend with non-blocking output collection."""

    def __init__(self) -> None:
        self._queues: Dict[int, Tuple[queue.Queue[str], queue.Queue[str]]] = {}

    def submit(self, request: TaskRequest) -> LocalHandle:
        cmd, run_shell = _build_command(request)
        env = _build_runtime_env(request)
        _validate_executable(request.command[0], env)

        process = subprocess.Popen(
            cmd,
            cwd=request.cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=run_shell,
            bufsize=1,
            start_new_session=True,
        )
        handle = LocalHandle(process=process)
        self._attach_readers(handle)
        return handle

    def poll(self, handle: LocalHandle) -> Tuple[bool, Optional[int]]:
        rc = handle.process.poll()
        return (rc is not None, rc)

    def read_incremental_output(self, handle: LocalHandle) -> Tuple[str, str]:
        qout, qerr = self._queues[handle.process.pid]
        out = _drain_queue(qout)
        err = _drain_queue(qerr)
        return out, err

    def cancel(self, handle: LocalHandle) -> None:
        if handle.process.poll() is not None:
            return
        try:
            os.killpg(handle.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    def cleanup(self, handle: LocalHandle) -> None:
        try:
            if handle.process.poll() is None:
                try:
                    os.killpg(handle.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            handle.process.wait(timeout=5)
        except Exception:
            pass
        self._queues.pop(handle.process.pid, None)

    def _attach_readers(self, handle: LocalHandle) -> None:
        qout: queue.Queue[str] = queue.Queue()
        qerr: queue.Queue[str] = queue.Queue()
        self._queues[handle.process.pid] = (qout, qerr)

        def pump(stream: Any, q: queue.Queue[str]) -> None:
            try:
                for line in iter(stream.readline, ""):
                    q.put(line)
            finally:
                stream.close()

        threading.Thread(target=pump, args=(handle.process.stdout, qout), daemon=True).start()
        threading.Thread(target=pump, args=(handle.process.stderr, qerr), daemon=True).start()


class EdaSubprocessTool:
    def __init__(self, backend: Optional[ExecutionBackend] = None, max_output_chars: int = 12000) -> None:
        self.backend = backend or LocalSubprocessBackend()
        self.max_output_chars = max_output_chars
        self._tasks: Dict[str, RunningTask] = {}
        self._handles: Dict[str, BackendHandle] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    @staticmethod
    def input_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["command"],
            "properties": {
                "command": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "cwd": {"type": "string"},
                "env": {"type": "object", "additionalProperties": {"type": "string"}},
                "login_shell": {"type": "boolean"},
                "shell_init_script": {"type": "string"},
                "timeout_seconds": {"type": "number", "minimum": 0},
                "retry_policy": {
                    "type": "object",
                    "properties": {
                        "max_retries": {"type": "integer", "minimum": 0},
                        "backoff_seconds": {"type": "number", "minimum": 0},
                    },
                },
                "preflight_checks": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                },
                "metadata": {"type": "object"},
            },
        }

    def submit(self, payload: Mapping[str, Any]) -> str:
        request = _parse_request(payload)
        task_id = str(uuid.uuid4())
        task = RunningTask(task_id=task_id, request=request)
        with self._lock:
            self._tasks[task_id] = task
        t = threading.Thread(target=self._run_task_lifecycle, args=(task_id,), daemon=True)
        with self._lock:
            self._threads[task_id] = t
        t.start()
        return task_id

    def run_sync(self, payload: Mapping[str, Any], poll_interval: float = 0.5) -> TaskResult:
        task_id = self.submit(payload)
        while True:
            result = self.get_result(task_id)
            with self._lock:
                worker = self._threads.get(task_id)
                worker_alive = bool(worker and worker.is_alive())
            if result.state in {TaskState.SUCCEEDED, TaskState.FAILED, TaskState.TIMEOUT, TaskState.CANCELLED}:
                if not worker_alive:
                    return result
            time.sleep(poll_interval)

    def cancel(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.cancel_requested = True
            handle = self._handles.get(task_id)
        if handle:
            self.backend.cancel(handle)

    def status(self, task_id: str) -> Dict[str, Any]:
        with self._lock:
            task = self._tasks[task_id]
            return {
                "task_id": task.task_id,
                "state": task.state.value,
                "attempt": task.attempt,
                "started_at": task.started_at,
                "ended_at": task.ended_at,
                "return_code": task.return_code,
                "backend_job_id": task.backend_job_id,
                "error_message": task.error_message,
            }

    def get_result(self, task_id: str) -> TaskResult:
        with self._lock:
            task = self._tasks[task_id]
            return TaskResult(
                task_id=task.task_id,
                state=task.state,
                return_code=task.return_code,
                started_at=task.started_at,
                ended_at=task.ended_at,
                attempt=task.attempt,
                stdout_tail=task.stdout_buffer,
                stderr_tail=task.stderr_buffer,
                error_message=task.error_message,
                backend_job_id=task.backend_job_id,
            )

    def _run_task_lifecycle(self, task_id: str) -> None:
        try:
            with self._lock:
                task = self._tasks[task_id]

            for attempt in range(task.request.retry_policy.max_retries + 1):
                with self._lock:
                    task.attempt = attempt
                    task.state = TaskState.PENDING
                    task.error_message = None
                try:
                    _run_preflight_checks(task.request)
                    self._execute_once(task)
                except Exception as exc:  # pylint: disable=broad-except
                    with self._lock:
                        task.state = TaskState.FAILED
                        task.error_message = str(exc)
                        task.ended_at = time.time()

                result = self.get_result(task_id)
                if result.state in {TaskState.SUCCEEDED, TaskState.CANCELLED}:
                    return
                if attempt < task.request.retry_policy.max_retries:
                    time.sleep(task.request.retry_policy.backoff_seconds)
                    continue
                return
        finally:
            with self._lock:
                self._threads.pop(task_id, None)

    def _execute_once(self, task: RunningTask) -> None:
        with self._lock:
            task.started_at = time.time()
            task.ended_at = None
            task.state = TaskState.RUNNING

        handle = self.backend.submit(task.request)
        with self._lock:
            self._handles[task.task_id] = handle
            task.backend_job_id = getattr(handle, "job_id", None)

        try:
            while True:
                if task.cancel_requested:
                    self.backend.cancel(handle)
                out, err = self.backend.read_incremental_output(handle)
                with self._lock:
                    task.stdout_buffer = _append_tail(task.stdout_buffer, out, self.max_output_chars)
                    task.stderr_buffer = _append_tail(task.stderr_buffer, err, self.max_output_chars)

                done, rc = self.backend.poll(handle)
                now = time.time()
                if done:
                    with self._lock:
                        task.return_code = rc
                        task.ended_at = now
                        if task.cancel_requested:
                            task.state = TaskState.CANCELLED
                        elif rc == 0:
                            task.state = TaskState.SUCCEEDED
                        else:
                            task.state = TaskState.FAILED
                    return

                if task.request.timeout_seconds is not None and task.started_at is not None:
                    if now - task.started_at > task.request.timeout_seconds:
                        self.backend.cancel(handle)
                        time.sleep(1.0)
                        self.backend.cleanup(handle)
                        with self._lock:
                            task.state = TaskState.TIMEOUT
                            task.error_message = f"Task timed out after {task.request.timeout_seconds}s"
                            task.ended_at = now
                        return
                time.sleep(0.4)
        finally:
            self.backend.cleanup(handle)
            with self._lock:
                self._handles.pop(task.task_id, None)


def _run_preflight_checks(request: TaskRequest) -> None:
    for check_cmd in request.preflight_checks:
        cmd = check_cmd
        run_shell = False
        if request.login_shell:
            cmd = ["bash", "-lc", shlex.join(check_cmd)]
        proc = subprocess.run(
            cmd,
            cwd=request.cwd,
            env=_build_runtime_env(request),
            capture_output=True,
            text=True,
            shell=run_shell,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Preflight check failed: "
                f"{check_cmd!r}, rc={proc.returncode}, stderr={proc.stderr.strip()}"
            )


def _parse_request(payload: Mapping[str, Any]) -> TaskRequest:
    if "command" not in payload:
        raise ValueError("'command' is required")
    retry_data = payload.get("retry_policy", {}) or {}
    request = TaskRequest(
        command=list(payload["command"]),
        cwd=payload.get("cwd"),
        env=dict(payload.get("env", {}) or {}),
        login_shell=bool(payload.get("login_shell", False)),
        shell_init_script=payload.get("shell_init_script"),
        timeout_seconds=payload.get("timeout_seconds"),
        retry_policy=RetryPolicy(
            max_retries=int(retry_data.get("max_retries", 0)),
            backoff_seconds=float(retry_data.get("backoff_seconds", 2.0)),
        ),
        preflight_checks=[list(x) for x in payload.get("preflight_checks", [])],
        metadata=dict(payload.get("metadata", {}) or {}),
    )
    if not request.command:
        raise ValueError("command cannot be empty")
    return request


def _build_runtime_env(request: TaskRequest) -> Dict[str, str]:
    env = dict(os.environ)
    if request.shell_init_script:
        sourced = _capture_env_from_script(request.shell_init_script)
        env.update(sourced)
    env.update(request.env)
    return env


def _capture_env_from_script(script_path: str) -> Dict[str, str]:
    script = Path(script_path)
    if not script.exists():
        raise FileNotFoundError(f"shell_init_script not found: {script_path}")
    cmd = ["bash", "-lc", f"source {shlex.quote(str(script))} >/dev/null 2>&1 && env -0"]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    out = proc.stdout.decode("utf-8", errors="replace")
    result: Dict[str, str] = {}
    for item in out.split("\x00"):
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        result[k] = v
    return result


def _build_command(request: TaskRequest) -> Tuple[List[str] | str, bool]:
    if request.login_shell:
        return ["bash", "-lc", shlex.join(request.command)], False
    return request.command, False


def _validate_executable(program: str, env: Mapping[str, str]) -> None:
    if os.path.isabs(program) or program.startswith("."):
        if not os.path.exists(program):
            raise FileNotFoundError(f"Executable not found: {program}")
        return

    path = env.get("PATH", os.environ.get("PATH", ""))
    for folder in path.split(os.pathsep):
        candidate = os.path.join(folder, program)
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return
    raise FileNotFoundError(f"Executable '{program}' not found in PATH")


def _append_tail(current: str, new_text: str, max_chars: int) -> str:
    merged = current + new_text
    if len(merged) <= max_chars:
        return merged
    return merged[-max_chars:]


def _drain_queue(q: queue.Queue[str]) -> str:
    chunks: List[str] = []
    while True:
        try:
            chunks.append(q.get_nowait())
        except queue.Empty:
            break
    return "".join(chunks)


if __name__ == "__main__":
    # tiny CLI demo for manual checks
    tool = EdaSubprocessTool()
    payload = {
        "command": ["python", "-c", "print('hello from EDA tool')"],
        "timeout_seconds": 10,
    }
    result = tool.run_sync(payload)
    print(json.dumps({
        "task_id": result.task_id,
        "state": result.state.value,
        "return_code": result.return_code,
        "stdout_tail": result.stdout_tail,
        "stderr_tail": result.stderr_tail,
    }, indent=2))
