"""配置管理工具。"""

import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "default_backend": "local",
    "backends": {
        "local": {
            "env_file": "~/.my_eda_env",
            "log_dir": "~/.eda_logs",
        }
    },
    "task_manager": {
        "max_parallel": 10,
        "poll_interval": 5.0,
        "state_file": "~/.eda_runner_state.json",
    },
}


def load_config(path: str = "~/.eda_runner.json") -> Dict[str, Any]:
    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        return DEFAULT_CONFIG.copy()
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)
