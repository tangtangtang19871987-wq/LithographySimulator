"""环境变量工具。"""

import os
from pathlib import Path
from typing import Dict


def read_env_file(path: str) -> Dict[str, str]:
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return {}

    env: Dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        if key:
            env[key] = value
    return env


def merged_env(extra: Dict[str, str]) -> Dict[str, str]:
    env = dict(os.environ)
    env.update(extra)
    return env
