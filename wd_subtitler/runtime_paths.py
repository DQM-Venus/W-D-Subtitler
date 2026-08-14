"""应用运行时路径的集中定义。"""

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
APP_DIR = PACKAGE_DIR.parent
if os.name == "nt":
    USER_DATA_DIR = Path(
        os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
    ) / "W-D Subtitler"
else:
    USER_DATA_DIR = Path(
        os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
    ) / "w-d-subtitler"
CONFIG_FILE = USER_DATA_DIR / "config.json"
ASSET_DIR = APP_DIR / "assets"
APP_ICON = ASSET_DIR / "app.ico"
LOG_DIR = APP_DIR / "logs"
CHECKPOINT_DIR = APP_DIR / "checkpoints"


def ensure_log_dir() -> Path:
    """创建并返回日志目录。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR
