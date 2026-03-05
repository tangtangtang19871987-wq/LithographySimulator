"""本地执行后端。"""

import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .base import Backend, StatusResult, SubmitResult, TaskResult, TaskStatus
from ..utils.env import load_env_file
from ..utils.logger import logger


class LocalBackend(Backend):
    def __init__(self, env_file: str = "~/.my_eda_env", log_dir: str = "~/.eda_logs"):
        self._env_file = Path(env_file).expanduser()
        self._log_dir = Path(log_dir).expanduser()

        if self._env_file.exists():
            self._env = load_env_file(str(self._env_file))
        else:
            logger.warning(f"环境文件不存在: {self._env_file}，使用当前环境")
            self._env = dict(os.environ)

        self._log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "local"

    def submit(self, cmd: str, task_id: str, startup_check: float = None, workdir: str = None, timeout: int = None) -> SubmitResult:
        task_dir = self._get_task_dir(task_id)
        try:
            task_dir.mkdir(parents=True, exist_ok=True)
            workdir = str(Path(workdir).expanduser()) if workdir else os.getcwd()

            meta = {
                "cmd": cmd,
                "task_id": task_id,
                "backend": self.name,
                "start_time": datetime.now().isoformat(),
                "timeout": timeout,
                "workdir": workdir,
            }
            self._write_meta(task_id, meta)

            stdout_log = task_dir / "stdout.log"
            stderr_log = task_dir / "stderr.log"
            exitcode_file = task_dir / "exitcode"

            wrapper = f"""
cd {workdir} 2>/dev/null || true
{cmd}
echo $? > {exitcode_file}
"""

            with open(stdout_log, "w", encoding="utf-8") as out, open(stderr_log, "w", encoding="utf-8") as err:
                proc = subprocess.Popen(
                    ["bash", "-c", wrapper],
                    stdout=out,
                    stderr=err,
                    env=self._env,
                    start_new_session=True,
                    cwd=workdir,
                )

            pid = proc.pid
            (task_dir / "pid").write_text(str(pid), encoding="utf-8")
            meta["pid"] = pid
            self._write_meta(task_id, meta)

            confirmed = False
            if startup_check and startup_check > 0:
                time.sleep(startup_check)
                ret = proc.poll()
                if ret is not None:
                    stderr_content = stderr_log.read_text(encoding="utf-8") if stderr_log.exists() else ""
                    return SubmitResult(ok=False, task_id=task_id, pid=pid, error="启动后立即退出", exitcode=ret, stderr=stderr_content, log_dir=str(task_dir))
                confirmed = True

            return SubmitResult(ok=True, task_id=task_id, pid=pid, confirmed=confirmed, log_dir=str(task_dir))
        except Exception as e:
            logger.exception("任务提交失败")
            return SubmitResult(ok=False, task_id=task_id, error=str(e), log_dir=str(task_dir))

    def check_status(self, task_id: str) -> StatusResult:
        task_dir = self._get_task_dir(task_id)
        if not task_dir.exists():
            return StatusResult(status=TaskStatus.UNKNOWN, error=f"任务目录不存在: {task_dir}")

        exitcode_file = task_dir / "exitcode"
        pid_file = task_dir / "pid"

        if exitcode_file.exists():
            try:
                exitcode = int(exitcode_file.read_text(encoding="utf-8").strip())
                return StatusResult(status=TaskStatus.SUCCESS if exitcode == 0 else TaskStatus.FAILED, exitcode=exitcode)
            except ValueError:
                return StatusResult(status=TaskStatus.UNKNOWN, error="无法解析 exitcode 文件")

        if not pid_file.exists():
            return StatusResult(status=TaskStatus.UNKNOWN, error="PID 文件不存在")

        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
            return StatusResult(status=TaskStatus.RUNNING)
        except ProcessLookupError:
            return StatusResult(status=TaskStatus.LOST, error="进程已消失但无 exitcode")
        except ValueError:
            return StatusResult(status=TaskStatus.UNKNOWN, error="无法解析 PID 文件")

    def get_result(self, task_id: str, **kwargs) -> TaskResult:
        task_dir = self._get_task_dir(task_id)
        stdout_file = task_dir / "stdout.log"
        stderr_file = task_dir / "stderr.log"
        exitcode_file = task_dir / "exitcode"

        stdout = stdout_file.read_text(encoding="utf-8") if stdout_file.exists() else ""
        stderr = stderr_file.read_text(encoding="utf-8") if stderr_file.exists() else ""
        exitcode = None
        if exitcode_file.exists():
            try:
                exitcode = int(exitcode_file.read_text(encoding="utf-8").strip())
            except ValueError:
                pass
        return TaskResult(stdout=stdout, stderr=stderr, exitcode=exitcode)

    def kill(self, task_id: str) -> bool:
        task_dir = self._get_task_dir(task_id)
        pid_file = task_dir / "pid"
        if not pid_file.exists():
            return False
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                return True
            time.sleep(1)
            try:
                os.kill(pid, 0)
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return True
        except Exception:
            logger.exception("终止任务失败")
            return False

    def cleanup(self, task_id: str) -> bool:
        task_dir = self._get_task_dir(task_id)
        if not task_dir.exists():
            return True
        try:
            shutil.rmtree(task_dir)
            return True
        except Exception:
            logger.exception("清理任务失败")
            return False

    def list_tasks(self) -> List[str]:
        tasks: List[str] = []
        if self._log_dir.exists():
            for item in self._log_dir.iterdir():
                if item.is_dir() and (item / "meta.json").exists():
                    tasks.append(item.name)
        return tasks

    def _get_task_dir(self, task_id: str) -> Path:
        return self._log_dir / task_id

    def _write_meta(self, task_id: str, meta: Dict):
        task_dir = self._get_task_dir(task_id)
        with open(task_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
