"""Socket API 后端。"""

import socket
import time
from typing import Any, Callable, Dict, Optional

from .base import Backend, StatusResult, SubmitResult, TaskResult, TaskStatus
from ..exceptions import ConnectionError
from ..protocols.base import ProtocolAdapter
from ..protocols.custom import CustomProtocol
from ..protocols.json_protocol import JSONProtocol
from ..protocols.text_protocol import TextProtocol


class SocketBackend(Backend):
    def __init__(self, host: str, port: int, timeout: float = 30.0, protocol: str = "json", persistent: bool = True, custom_encoder: Callable = None, custom_decoder: Callable = None, recv_terminator: bytes = b"\n", recv_buffer_size: int = 4096):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._persistent = persistent
        self._recv_terminator = recv_terminator
        self._recv_buffer_size = recv_buffer_size
        self._socket: Optional[socket.socket] = None

        if protocol == "json":
            self._protocol: ProtocolAdapter = JSONProtocol()
        elif protocol == "text":
            self._protocol = TextProtocol()
        elif protocol == "custom":
            if not custom_encoder or not custom_decoder:
                raise ValueError("custom 协议需要提供 encoder 和 decoder")
            self._protocol = CustomProtocol(custom_encoder, custom_decoder)
        else:
            raise ValueError(f"未知协议: {protocol}")

    @property
    def name(self) -> str:
        return "socket"

    def connect(self) -> bool:
        if self._socket:
            return True
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self._timeout)
            self._socket.connect((self._host, self._port))
            return True
        except Exception as e:
            self._socket = None
            raise ConnectionError(f"连接失败: {e}")

    def disconnect(self):
        if self._socket:
            self._socket.close()
            self._socket = None

    def submit(self, cmd: str, task_id: str, wait_response: bool = True, response_timeout: float = None, **params) -> SubmitResult:
        try:
            self._ensure_connected()
            data = self._protocol.encode(cmd, task_id, **params)
            if not wait_response:
                self._send(data)
                return SubmitResult(ok=True, task_id=task_id, extra={"sent": True, "wait_response": False})
            response = self._protocol.decode(self._send_recv(data, response_timeout or self._timeout))
            ok = response.get("ok", response.get("status") == "ok")
            return SubmitResult(ok=ok, task_id=task_id, error=response.get("error"), extra={"response": response})
        except Exception as e:
            return SubmitResult(ok=False, task_id=task_id, error=str(e))

    def check_status(self, task_id: str) -> StatusResult:
        try:
            self._ensure_connected()
            response = self._protocol.decode(self._send_recv(self._protocol.encode("status", task_id)))
            status_map = {"running": TaskStatus.RUNNING, "pending": TaskStatus.PENDING, "done": TaskStatus.SUCCESS, "success": TaskStatus.SUCCESS, "ok": TaskStatus.SUCCESS, "failed": TaskStatus.FAILED, "error": TaskStatus.FAILED}
            status = status_map.get(str(response.get("status", "unknown")).lower(), TaskStatus.UNKNOWN)
            return StatusResult(status=status, exitcode=response.get("exitcode"), raw=response)
        except Exception as e:
            return StatusResult(status=TaskStatus.UNKNOWN, error=str(e))

    def get_result(self, task_id: str, **kwargs) -> TaskResult:
        try:
            self._ensure_connected()
            response = self._protocol.decode(self._send_recv(self._protocol.encode("result", task_id)))
            return TaskResult(stdout=response.get("stdout", ""), stderr=response.get("stderr", ""), exitcode=response.get("exitcode"), data=response.get("data"))
        except Exception as e:
            return TaskResult(stderr=str(e))

    def kill(self, task_id: str) -> bool:
        try:
            self._ensure_connected()
            response = self._protocol.decode(self._send_recv(self._protocol.encode("kill", task_id)))
            return response.get("ok", False)
        except Exception:
            return False

    def _ensure_connected(self):
        if not self._socket:
            self.connect()

    def _send(self, data: bytes):
        self._socket.sendall(data)

    def _recv(self, timeout: float = None) -> bytes:
        if timeout:
            self._socket.settimeout(timeout)
        response = b""
        while True:
            chunk = self._socket.recv(self._recv_buffer_size)
            if not chunk:
                break
            response += chunk
            if response.endswith(self._recv_terminator):
                break
        return response

    def _send_recv(self, data: bytes, timeout: float = None) -> bytes:
        self._send(data)
        response = self._recv(timeout)
        if not self._persistent:
            self.disconnect()
        return response
