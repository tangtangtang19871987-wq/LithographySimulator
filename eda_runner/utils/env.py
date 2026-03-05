"""环境变量加载工具。"""

import os
import re
from pathlib import Path
from typing import Dict


def load_env_file(env_file: str) -> Dict[str, str]:
    path = Path(env_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"环境文件不存在: {path}")

    env: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line.strip() or line.strip().startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or " " in key or "\t" in key:
                continue
            env[key] = value
    return env


def merge_env(base: Dict[str, str], override: Dict[str, str]) -> Dict[str, str]:
    result = base.copy()
    result.update(override)
    return result


def get_current_env() -> Dict[str, str]:
    return dict(os.environ)


def expand_env_vars(value: str, env: Dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        var_name = match.group(1) or match.group(2)
        return env.get(var_name, match.group(0))

    pattern = r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)"
    return re.sub(pattern, replace, value)
