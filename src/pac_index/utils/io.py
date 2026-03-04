"""File I/O and result serialization utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data: Any, path: str | Path) -> None:
    """Save data to a JSON file with NumPy type support."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def load_json(path: str | Path) -> Any:
    """Load data from a JSON file."""
    with open(path) as f:
        return json.load(f)


def save_yaml(data: Any, path: str | Path) -> None:
    """Save data to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_yaml(path: str | Path) -> Any:
    """Load data from a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Save a list of dictionaries to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: str | Path) -> list[dict[str, str]]:
    """Load a CSV file into a list of dictionaries."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_sorted_keys(path: str | Path, max_keys: int | None = None) -> np.ndarray:
    """Load sorted keys from a binary file (SOSD format: uint64 little-endian)."""
    data = np.fromfile(str(path), dtype=np.uint64)
    if max_keys is not None:
        data = data[:max_keys]
    return np.sort(data)
