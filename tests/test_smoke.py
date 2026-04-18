from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    os.environ.get("DIAGNOSTICIAN_RUN_DOCKER_SMOKE") != "1",
    reason="Set DIAGNOSTICIAN_RUN_DOCKER_SMOKE=1 to run the docker compose smoke test.",
)
def test_docker_compose_stack_smoke() -> None:
    subprocess.run(
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "smoke-compose.ps1"),
        ],
        cwd=ROOT,
        check=True,
        timeout=240,
    )
