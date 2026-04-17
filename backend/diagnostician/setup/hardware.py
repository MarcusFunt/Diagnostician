from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

import httpx


DEFAULT_DATABASE_URL = "postgresql+psycopg://diagnostician:diagnostician@localhost:5432/diagnostician"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MEDICAL_CHECK_MODEL = "hf.co/tensorblock/Llama3-Med42-8B-GGUF"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

CommandRunner = Callable[[Sequence[str], int], str]


@dataclass(frozen=True)
class HardwareScan:
    total_ram_gb: float
    available_ram_gb: float | None = None
    cpu_name: str | None = None
    cpu_cores: int | None = None
    gpu_names: list[str] = field(default_factory=list)
    gpu_memory_gb: list[float] = field(default_factory=list)
    nvidia_vram_gb: list[float] = field(default_factory=list)
    ollama_ok: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def max_vram_gb(self) -> float:
        values = [*self.gpu_memory_gb, *self.nvidia_vram_gb]
        return max(values) if values else 0.0


@dataclass(frozen=True)
class ModelSelection:
    case_generator_model: str
    medical_check_model: str = DEFAULT_MEDICAL_CHECK_MODEL
    medical_check_enabled: bool = True
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ollama_keep_alive: str = "0"
    warnings: list[str] = field(default_factory=list)


def scan_hardware(
    *,
    run_command: CommandRunner | None = None,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> HardwareScan:
    runner = run_command or _run_command
    warnings: list[str] = []
    total_ram_gb, available_ram_gb = _memory_gb(runner, warnings)
    cpu_name, cpu_cores = _cpu_info(runner, warnings)
    gpu_names, gpu_memory_gb = _gpu_info(runner, warnings)
    nvidia_vram_gb = _nvidia_vram_gb(runner)
    ollama_ok = _ollama_reachable(ollama_base_url)
    if not ollama_ok:
        warnings.append(f"Ollama was not reachable at {ollama_base_url}.")
    return HardwareScan(
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        gpu_names=gpu_names,
        gpu_memory_gb=gpu_memory_gb,
        nvidia_vram_gb=nvidia_vram_gb,
        ollama_ok=ollama_ok,
        warnings=warnings,
    )


def select_models(
    scan: HardwareScan,
    *,
    medical_check_model: str = DEFAULT_MEDICAL_CHECK_MODEL,
) -> ModelSelection:
    ram = scan.total_ram_gb
    vram = scan.max_vram_gb
    warnings = list(scan.warnings)

    if ram >= 48 or vram >= 16:
        qwen_model = "qwen3:30b"
    elif ram >= 32 or vram >= 12:
        qwen_model = "qwen3:14b"
    elif ram >= 24 or vram >= 8:
        qwen_model = "qwen3:8b"
    elif ram < 12 and vram < 8:
        qwen_model = "qwen3:1.7b"
        warnings.append(
            "Less than 12 GB RAM detected; Med42 checks may be slow or unreliable locally."
        )
    else:
        qwen_model = "qwen3:4b-instruct"

    limited_hardware = ram < 24 and vram < 8
    medical_check_enabled = not limited_hardware
    keep_alive = "5m"
    if not medical_check_enabled:
        warnings.append(
            "Limited RAM/VRAM detected; Med42 checks will be disabled locally and gameplay will rely on Qwen plus deterministic validation."
        )
    return ModelSelection(
        case_generator_model=qwen_model,
        medical_check_model=medical_check_model,
        medical_check_enabled=medical_check_enabled,
        ollama_keep_alive=keep_alive,
        warnings=warnings,
    )


def build_env_values(
    selection: ModelSelection,
    *,
    database_url: str = DEFAULT_DATABASE_URL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> dict[str, str]:
    return {
        "DIAGNOSTICIAN_DATABASE_URL": database_url,
        "DIAGNOSTICIAN_OLLAMA_BASE_URL": ollama_base_url,
        "DIAGNOSTICIAN_GENERATION_MODEL": selection.case_generator_model,
        "DIAGNOSTICIAN_CASE_GENERATOR_MODEL": selection.case_generator_model,
        "DIAGNOSTICIAN_MEDICAL_CHECK_MODEL": selection.medical_check_model,
        "DIAGNOSTICIAN_MEDICAL_CHECK_ENABLED": "true" if selection.medical_check_enabled else "false",
        "DIAGNOSTICIAN_EMBEDDING_MODEL": selection.embedding_model,
        "DIAGNOSTICIAN_REQUIRE_OLLAMA": "false",
        "DIAGNOSTICIAN_EMBEDDING_DIMENSIONS": "768",
        "DIAGNOSTICIAN_OLLAMA_KEEP_ALIVE": selection.ollama_keep_alive,
        "DIAGNOSTICIAN_GENERATION_REPAIR_ATTEMPTS": "2",
        "DIAGNOSTICIAN_LLM_TIMEOUT_SECONDS": "600",
    }


def write_env_file(env_path: str | Path, values: dict[str, str]) -> Path:
    path = Path(env_path)
    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    consumed: set[str] = set()
    next_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            next_lines.append(line)
            continue
        key, _value = line.split("=", 1)
        if key in values:
            next_lines.append(f"{key}={values[key]}")
            consumed.add(key)
        else:
            next_lines.append(line)
    for key, value in values.items():
        if key not in consumed:
            next_lines.append(f"{key}={value}")
    path.write_text("\n".join(next_lines).rstrip() + "\n", encoding="utf-8")
    return path


def payload_for(scan: HardwareScan, selection: ModelSelection, env_path: str | Path | None = None) -> dict[str, Any]:
    return {
        "scan": {**asdict(scan), "max_vram_gb": scan.max_vram_gb},
        "selection": asdict(selection),
        "env_path": str(env_path) if env_path is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan hardware and configure local Diagnostician models.")
    parser.add_argument("--write-env", action="store_true", help="Write selected settings to an env file.")
    parser.add_argument("--env-path", default=".env", help="Env file to create or update.")
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    parser.add_argument("--medical-check-model", default=DEFAULT_MEDICAL_CHECK_MODEL)
    parser.add_argument("--json", action="store_true", help="Print machine-readable scan and selection data.")
    args = parser.parse_args()

    scan = scan_hardware(ollama_base_url=args.ollama_base_url)
    selection = select_models(scan, medical_check_model=args.medical_check_model)
    env_path = Path(args.env_path)
    if args.write_env:
        write_env_file(
            env_path,
            build_env_values(
                selection,
                database_url=args.database_url,
                ollama_base_url=args.ollama_base_url,
            ),
        )

    payload = payload_for(scan, selection, env_path if args.write_env else None)
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Selected Qwen model: {selection.case_generator_model}")
    print(f"Selected medical checker: {selection.medical_check_model}")
    print(f"Ollama keep_alive: {selection.ollama_keep_alive}")
    if args.write_env:
        print(f"Updated {env_path}")
    for warning in selection.warnings:
        print(f"warning: {warning}")


def _run_command(command: Sequence[str], timeout: int) -> str:
    completed = subprocess.run(
        list(command),
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip())
    return completed.stdout.strip()


def _memory_gb(runner: CommandRunner, warnings: list[str]) -> tuple[float, float | None]:
    if platform.system().casefold() == "windows":
        try:
            system_data = _powershell_json(
                runner,
                "Get-CimInstance Win32_ComputerSystem | "
                "Select-Object TotalPhysicalMemory | ConvertTo-Json -Compress",
            )
            os_data = _powershell_json(
                runner,
                "Get-CimInstance Win32_OperatingSystem | "
                "Select-Object FreePhysicalMemory | ConvertTo-Json -Compress",
            )
            total = _bytes_to_gb(float(system_data.get("TotalPhysicalMemory") or 0))
            free_kb = float(os_data.get("FreePhysicalMemory") or 0)
            return total, free_kb / 1024 / 1024
        except Exception as exc:
            warnings.append(f"Windows memory scan failed: {exc}")

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return _bytes_to_gb(float(pages * page_size)), None
    except Exception as exc:
        warnings.append(f"Memory scan failed: {exc}")
        return 0.0, None


def _cpu_info(runner: CommandRunner, warnings: list[str]) -> tuple[str | None, int | None]:
    if platform.system().casefold() == "windows":
        try:
            data = _powershell_json(
                runner,
                "Get-CimInstance Win32_Processor | "
                "Select-Object -First 1 Name,NumberOfCores | ConvertTo-Json -Compress",
            )
            return _clean_str(data.get("Name")), _clean_int(data.get("NumberOfCores"))
        except Exception as exc:
            warnings.append(f"Windows CPU scan failed: {exc}")
    return platform.processor() or None, os.cpu_count()


def _gpu_info(runner: CommandRunner, warnings: list[str]) -> tuple[list[str], list[float]]:
    if platform.system().casefold() != "windows":
        return [], []
    try:
        data = _powershell_json(
            runner,
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
        )
    except Exception as exc:
        warnings.append(f"Windows GPU scan failed: {exc}")
        return [], []
    rows = data if isinstance(data, list) else [data]
    names: list[str] = []
    memory: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = _clean_str(row.get("Name"))
        if name:
            names.append(name)
        adapter_ram = _clean_int(row.get("AdapterRAM"))
        if adapter_ram and adapter_ram > 0:
            memory.append(_bytes_to_gb(float(adapter_ram)))
    return names, memory


def _nvidia_vram_gb(runner: CommandRunner) -> list[float]:
    try:
        output = runner(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            5,
        )
    except Exception:
        return []
    values: list[float] = []
    for line in output.splitlines():
        try:
            values.append(float(line.strip()) / 1024)
        except ValueError:
            continue
    return values


def _ollama_reachable(base_url: str) -> bool:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def _powershell_json(runner: CommandRunner, command: str) -> Any:
    output = runner(["powershell", "-NoProfile", "-Command", command], 10)
    return json.loads(output or "{}")


def _bytes_to_gb(value: float) -> float:
    return round(value / 1024 / 1024 / 1024, 2)


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
