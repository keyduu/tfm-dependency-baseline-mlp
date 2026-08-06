"""Runtime provenance captured with reproducible experiment reports."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def runtime_environment(*, device: torch.device, libraries: dict[str, str]) -> dict[str, Any]:
    """Describe the host, process, accelerator and software used for timing."""

    environment: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "python": platform.python_version(),
        **libraries,
        "system": {
            "os": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "cpu": {
            "model": _cpu_model(),
            "logical_cores": os.cpu_count(),
            "available_logical_cores": _available_cpu_count(),
        },
        "memory": {"total_bytes": _memory_total_bytes()},
        "process": {
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "torch_num_threads": torch.get_num_threads(),
        },
        "graphics_devices": _graphics_devices(),
        "cuda": _cuda_metadata(device),
    }
    driver = _nvidia_driver_version()
    if driver is not None:
        environment["nvidia_driver"] = driver
    return environment


def synchronize(device: torch.device) -> None:
    """Wait for queued accelerator work before reading a wall-clock timer."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_metadata(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "device_index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return result


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    value = platform.processor().strip()
    return value or None


def _available_cpu_count() -> int | None:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def _memory_total_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB$", meminfo.read_text(), re.MULTILINE)
    return int(match.group(1)) * 1024 if match else None


def _graphics_devices() -> list[str]:
    executable = shutil.which("lspci")
    if executable is None:
        return []
    try:
        result = subprocess.run(
            [executable, "-mm"], check=False, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if '"VGA compatible controller"' in line or '"3D controller"' in line
    ]


def _nvidia_driver_version() -> str | None:
    version_file = Path("/proc/driver/nvidia/version")
    if not version_file.is_file():
        return None
    match = re.search(r"Kernel Module.*?\s([0-9]+(?:\.[0-9]+)+)\s", version_file.read_text())
    return match.group(1) if match else None
