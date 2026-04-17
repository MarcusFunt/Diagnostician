from __future__ import annotations

import os
import platform


def disable_slow_wmi_platform_probe() -> None:
    """Avoid Windows WMI hangs during dependency import-time platform checks."""

    if os.name == "nt" and hasattr(platform, "_wmi"):
        platform._wmi = None
