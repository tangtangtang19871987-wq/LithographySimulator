"""JSON 协议适配器。"""

import json
from typing import Any, Dict
from .base import ProtocolAdapter


class JSONProtocol(ProtocolAdapter):
    def encode(self, cmd: str, task_id: str, **kwargs) -> bytes:
        msg = {"cmd": cmd, "task_id": task_id, **kwargs}
        return (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")

    def decode(self, data: bytes) -> Dict[str, Any]:
        return json.loads(data.decode("utf-8").strip())
