"""Call quiet_framework_logs before importing the rest of openJiuwen, which logs while it loads."""
from __future__ import annotations

from pathlib import Path

from openjiuwen.core.common.logging.log_config import configure_log_config


def quiet_framework_logs(work_root: Path) -> None:
    """openJiuwen logs every request at INFO and warns about optional modules; keep only errors, on the console."""
    configure_log_config({
        "backend": "default",
        "level": "ERROR",
        "propagate": False,
        "log_path": str(work_root / "logs"),
        "output": ["console"],
        "interface_output": ["console"],
        "performance_output": ["console"],
    })
