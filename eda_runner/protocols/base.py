"""协议适配器基类。"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class ProtocolAdapter(ABC):
    @abstractmethod
    def encode(self, cmd: str, task_id: str, **kwargs) -> bytes:
        pass

    @abstractmethod
    def decode(self, data: bytes) -> Dict[str, Any]:
        pass
