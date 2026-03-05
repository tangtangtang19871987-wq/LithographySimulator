"""LSF 集群后端。"""

import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .base import Backend, StatusResult, SubmitResult, TaskResult, TaskStatus
from ..utils.env import load_env_file
from ..utils.logger import logger


class LSFBackend(Backend):
    def __init__(self, env_file: str = "~/.my_eda_env", log_dir: str = "~/.eda_logs", queue: str = None, resource: str = None):
        self._env_file = Path(env_file).expanduser()
        self._log_dir = Path(log_dir).expanduser()
        self._default_queue = queue
        self._default_resource = resource
        self._env = load_env_file(str(self._env_file)) if self._env_file.exists() else dict(os.environ)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "lsf"

    def submit(self, cmd: str, task_id: str, queue: str = None, resource: str = None, n_cores: int = 1, walltime: str = None, workdir: str = None) -> SubmitResult:
        task_dir = self._log_dir / task_id
        try:
            task_dir.mkdir(parents=True, exist_ok=True)
            queue = queue or self._default_queue
            resource = resource or self._default_resource
            workdir = str(Path(workdir).expanduser()) if workdir else os.getcwd()
            stdout_log = task_dir / "stdout.log"
            stderr_log = task_dir / "stderr.log"
            bsub_cmd = ["bsub"]
            if queue:
                bsub_cmd.extend(["-q", queue])
            if resource:
                bsub_cmd.extend(["-R", resource])
            bsub_cmd.extend(["-n", str(n_cores)])
            if walltime:
                bsub_cmd.extend(["-W", walltime])
            bsub_cmd.extend(["-o", str(stdout_log), "-e", str(stderr_log), "-J", task_id, "-cwd", workdir])
            bsub_cmd.append(f"source {self._env_file} && {cmd}")
            result = subprocess.run(bsub_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return SubmitResult(ok=False, task_id=task_id, error=f"bsub 失败: {result.stderr}", log_dir=str(task_dir))
            job_id = self._parse_job_id(result.stdout)
            if not job_id:
                return SubmitResult(ok=False, task_id=task_id, error=f"无法解析 job_id: {result.stdout}", log_dir=str(task_dir))
            (task_dir / "job_id").write_text(job_id, encoding="utf-8")
            meta = {"cmd": cmd, "task_id": task_id, "backend": self.name, "start_time": datetime.now().isoformat(), "job_id": job_id, "queue": queue, "resource": resource, "n_cores": n_cores, "walltime": walltime, "workdir": workdir}
            with open(task_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            return SubmitResult(ok=True, task_id=task_id, job_id=job_id, confirmed=True, log_dir=str(task_dir))
        except Exception as e:
            return SubmitResult(ok=False, task_id=task_id, error=str(e), log_dir=str(task_dir))

    def check_status(self, task_id: str) -> StatusResult:
        job_id_file = self._log_dir / task_id / "job_id"
        if not job_id_file.exists():
            return StatusResult(status=TaskStatus.UNKNOWN, error="Job ID 文件不存在")
        job_id = job_id_file.read_text(encoding="utf-8").strip()
        try:
            result = subprocess.run(["bjobs", "-o", "stat", "-noheader", job_id], capture_output=True, text=True, timeout=10)
            stat = result.stdout.strip()
            mapping = {"PEND": TaskStatus.PENDING, "RUN": TaskStatus.RUNNING, "DONE": TaskStatus.SUCCESS, "EXIT": TaskStatus.FAILED, "USUSP": TaskStatus.PENDING, "SSUSP": TaskStatus.PENDING, "PSUSP": TaskStatus.PENDING, "ZOMBI": TaskStatus.LOST}
            if stat in mapping:
                status = mapping[stat]
                exitcode = self._get_exitcode(job_id) if status in [TaskStatus.SUCCESS, TaskStatus.FAILED] else None
                return StatusResult(status=status, exitcode=exitcode, raw={"lsf_stat": stat})
            return StatusResult(status=TaskStatus.UNKNOWN, error="任务不在 LSF 队列中" if not stat else None, raw={"lsf_stat": stat} if stat else None)
        except Exception as e:
            return StatusResult(status=TaskStatus.UNKNOWN, error=str(e))

    def get_result(self, task_id: str, **kwargs) -> TaskResult:
        task_dir = self._log_dir / task_id
        stdout = (task_dir / "stdout.log").read_text(encoding="utf-8") if (task_dir / "stdout.log").exists() else ""
        stderr = (task_dir / "stderr.log").read_text(encoding="utf-8") if (task_dir / "stderr.log").exists() else ""
        exitcode = None
        if (task_dir / "job_id").exists():
            exitcode = self._get_exitcode((task_dir / "job_id").read_text(encoding="utf-8").strip())
        return TaskResult(stdout=stdout, stderr=stderr, exitcode=exitcode)

    def kill(self, task_id: str) -> bool:
        job_id_file = self._log_dir / task_id / "job_id"
        if not job_id_file.exists():
            return False
        job_id = job_id_file.read_text(encoding="utf-8").strip()
        try:
            result = subprocess.run(["bkill", job_id], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False

    def cleanup(self, task_id: str) -> bool:
        task_dir = self._log_dir / task_id
        if not task_dir.exists():
            return True
        try:
            shutil.rmtree(task_dir)
            return True
        except Exception:
            return False

    def list_tasks(self) -> List[str]:
        return [i.name for i in self._log_dir.iterdir() if i.is_dir() and (i / "job_id").exists()] if self._log_dir.exists() else []

    def _parse_job_id(self, bsub_output: str) -> Optional[str]:
        m = re.search(r"Job <(\d+)>", bsub_output)
        return m.group(1) if m else None

    def _get_exitcode(self, job_id: str) -> Optional[int]:
        try:
            result = subprocess.run(["bjobs", "-o", "exit_code", "-noheader", job_id], capture_output=True, text=True, timeout=10)
            code_str = result.stdout.strip()
            if code_str and code_str != "-":
                return int(code_str)
        except Exception:
            pass
        return None
