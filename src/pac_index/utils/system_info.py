"""Hardware and OS information capture for experiment reproducibility."""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any

import psutil


def get_cpu_info() -> dict[str, Any]:
    """Get detailed CPU information."""
    info: dict[str, Any] = {
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "max_frequency_mhz": None,
        "architecture": platform.machine(),
    }
    freq = psutil.cpu_freq()
    if freq:
        info["max_frequency_mhz"] = freq.max
    return info


def get_memory_info() -> dict[str, Any]:
    """Get memory information."""
    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / (1024**3), 1),
        "available_gb": round(mem.available / (1024**3), 1),
    }


def get_os_info() -> dict[str, str]:
    """Get operating system information."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "platform": platform.platform(),
    }


def get_python_info() -> dict[str, str]:
    """Get Python environment information."""
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler(),
    }


def get_full_system_info() -> dict[str, Any]:
    """Collect comprehensive system information for reproducibility."""
    return {
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "os": get_os_info(),
        "python": get_python_info(),
    }


def print_system_info() -> None:
    """Print system information to stdout."""
    info = get_full_system_info()
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    for category, details in info.items():
        print(f"\n[{category.upper()}]")
        if isinstance(details, dict):
            for key, value in details.items():
                print(f"  {key}: {value}")
    print("=" * 60)
