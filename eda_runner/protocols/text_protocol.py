"""文本协议适配器。"""

from typing import Any, Dict
from .base import ProtocolAdapter


class TextProtocol(ProtocolAdapter):
    def encode(self, cmd: str, task_id: str, **kwargs) -> bytes:
        params = " ".join(f"{k}={v}" for k, v in kwargs.items())
        line = f"{cmd} task_id={task_id} {params}".rstrip() + "\n"
        return line.encode("utf-8")

    def decode(self, data: bytes) -> Dict[str, Any]:
        text = data.decode("utf-8").strip()
        if text.startswith("OK:"):
            return {"ok": True, "message": text[3:].strip()}
        if text.startswith("ERROR:"):
            return {"ok": False, "error": text[6:].strip()}
        return {"ok": True, "raw": text}
