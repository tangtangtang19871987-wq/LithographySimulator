"""SSH 远程后端。"""

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .base import Backend, StatusResult, SubmitResult, TaskResult, TaskStatus


class SSHBackend(Backend):
    def __init__(self, host: str, user: str = None, port: int = 22, key_file: str = None, env_file: str = "~/.my_eda_env", remote_log_dir: str = "~/.eda_logs", local_log_dir: str = "~/.eda_logs_local", ssh_options: Dict[str, str] = None):
        self._host = host
        self._user = user or os.environ.get("USER", "root")
        self._port = port
        self._key_file = Path(key_file).expanduser() if key_file else None
        self._env_file = env_file
        self._remote_log_dir = remote_log_dir
        self._local_log_dir = Path(local_log_dir).expanduser()
        self._local_log_dir.mkdir(parents=True, exist_ok=True)
        self._ssh_options = {"BatchMode": "yes", "StrictHostKeyChecking": "no", "ConnectTimeout": "10"}
        if ssh_options:
            self._ssh_options.update(ssh_options)

    @property
    def name(self) -> str:
        return "ssh"

    def submit(self, cmd: str, task_id: str, startup_check: float = None, workdir: str = None, timeout: int = None) -> SubmitResult:
        local_task_dir = self._local_log_dir / task_id
        remote_task_dir = f"{self._remote_log_dir}/{task_id}"
        local_task_dir.mkdir(parents=True, exist_ok=True)
        workdir = workdir or "~"
        remote_cmd = f"""
mkdir -p {remote_task_dir}
cd {workdir} 2>/dev/null || cd ~
source {self._env_file} 2>/dev/null || true
nohup bash -c '{cmd}; echo $? > {remote_task_dir}/exitcode' > {remote_task_dir}/stdout.log 2> {remote_task_dir}/stderr.log &
echo $! > {remote_task_dir}/pid
cat {remote_task_dir}/pid
"""
        try:
            result = self._ssh_cmd(remote_cmd, timeout=30)
            if result.returncode != 0:
                return SubmitResult(ok=False, task_id=task_id, error=result.stderr, log_dir=str(local_task_dir))
            pid = int(result.stdout.strip().split("\n")[-1])
            meta = {"cmd": cmd, "task_id": task_id, "backend": self.name, "start_time": datetime.now().isoformat(), "host": self._host, "remote_pid": pid, "remote_log_dir": remote_task_dir, "workdir": workdir, "timeout": timeout}
            with open(local_task_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            return SubmitResult(ok=True, task_id=task_id, pid=pid, log_dir=str(local_task_dir), extra={"host": self._host, "remote_log_dir": remote_task_dir})
        except Exception as e:
            return SubmitResult(ok=False, task_id=task_id, error=str(e), log_dir=str(local_task_dir))

    def check_status(self, task_id: str) -> StatusResult:
        remote_task_dir = f"{self._remote_log_dir}/{task_id}"
        cmd = f"""
if [ -f {remote_task_dir}/exitcode ]; then
 echo DONE
 cat {remote_task_dir}/exitcode
else
 pid=$(cat {remote_task_dir}/pid 2>/dev/null)
 if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
   echo RUNNING
 else
   echo LOST
 fi
fi
"""
        try:
            result = self._ssh_cmd(cmd, timeout=10)
            if result.returncode != 0:
                return StatusResult(status=TaskStatus.UNKNOWN, error=result.stderr)
            lines = result.stdout.strip().split("\n")
            if lines[0] == "DONE":
                code = int(lines[1]) if len(lines) > 1 else None
                return StatusResult(status=TaskStatus.SUCCESS if code == 0 else TaskStatus.FAILED, exitcode=code)
            if lines[0] == "RUNNING":
                return StatusResult(status=TaskStatus.RUNNING)
            return StatusResult(status=TaskStatus.LOST)
        except Exception as e:
            return StatusResult(status=TaskStatus.UNKNOWN, error=str(e))

    def get_result(self, task_id: str, fetch: bool = False, **kwargs) -> TaskResult:
        remote_task_dir = f"{self._remote_log_dir}/{task_id}"
        cmd = f"""
echo "===STDOUT==="
cat {remote_task_dir}/stdout.log 2>/dev/null || true
echo "===STDERR==="
cat {remote_task_dir}/stderr.log 2>/dev/null || true
echo "===EXITCODE==="
cat {remote_task_dir}/exitcode 2>/dev/null || echo "NONE"
"""
        try:
            result = self._ssh_cmd(cmd, timeout=30)
            output = result.stdout
            stdout = output.split("===STDOUT===")[1].split("===STDERR===")[0].strip() if "===STDOUT===" in output and "===STDERR===" in output else ""
            stderr = output.split("===STDERR===")[1].split("===EXITCODE===")[0].strip() if "===STDERR===" in output and "===EXITCODE===" in output else ""
            exit_str = output.split("===EXITCODE===")[-1].strip() if "===EXITCODE===" in output else "NONE"
            exitcode = int(exit_str) if exit_str not in ["", "NONE"] and exit_str.isdigit() else None
            return TaskResult(stdout=stdout, stderr=stderr, exitcode=exitcode)
        except Exception as e:
            return TaskResult(stderr=str(e))

    def kill(self, task_id: str) -> bool:
        remote_task_dir = f"{self._remote_log_dir}/{task_id}"
        cmd = f"pid=$(cat {remote_task_dir}/pid 2>/dev/null); if [ -n \"$pid\" ]; then kill -9 $pid 2>/dev/null; echo OK; else echo NO_PID; fi"
        try:
            return "OK" in self._ssh_cmd(cmd, timeout=10).stdout
        except Exception:
            return False

    def cleanup(self, task_id: str) -> bool:
        ok = True
        local = self._local_log_dir / task_id
        if local.exists():
            try:
                shutil.rmtree(local)
            except Exception:
                ok = False
        try:
            self._ssh_cmd(f"rm -rf {self._remote_log_dir}/{task_id}", timeout=10)
        except Exception:
            ok = False
        return ok

    def _build_ssh_cmd(self) -> List[str]:
        cmd = ["ssh"]
        for k, v in self._ssh_options.items():
            cmd.extend(["-o", f"{k}={v}"])
        cmd.extend(["-p", str(self._port)])
        if self._key_file and self._key_file.exists():
            cmd.extend(["-i", str(self._key_file)])
        cmd.append(f"{self._user}@{self._host}")
        return cmd

    def _ssh_cmd(self, remote_cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
        cmd = self._build_ssh_cmd()
        cmd.append(remote_cmd)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
