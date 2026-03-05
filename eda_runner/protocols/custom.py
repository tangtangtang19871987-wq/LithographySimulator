"""自定义协议适配器。"""

from typing import Any, Callable, Dict
from .base import ProtocolAdapter


class CustomProtocol(ProtocolAdapter):
    def __init__(self, encode_fn: Callable, decode_fn: Callable):
        self._encode = encode_fn
        self._decode = decode_fn

    def encode(self, cmd: str, task_id: str, **kwargs) -> bytes:
        return self._encode(cmd, task_id, **kwargs)

    def decode(self, data: bytes) -> Dict[str, Any]:
        return self._decode(data)
