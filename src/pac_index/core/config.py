"""Configuration management for PAC-Index experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""

    id: str
    name: str
    source: str
    size: int
    key_type: str = "uint64"
    cdf_type: str = "smooth"
    cv: float = 0.0
    skewness: float = 0.0
    gap_autocorrelation: float = 0.0
    kurtosis: float = 0.0


@dataclass
class IndexConfig:
    """Configuration for an index structure."""

    name: str
    error_bound: int = 64
    levels: int = 2
    branching_factor: int = 100
    radix_bits: int = 18
    config: str = "default"


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    name: str = "default"
    seeds: list[int] = field(default_factory=lambda: [42, 123, 256, 314, 512, 628, 729, 841, 953, 1024])
    num_runs: int = 10
    warmup_runs: int = 3
    sample_sizes: list[int] = field(
        default_factory=lambda: [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    )
    cache_mode: str = "warm"


@dataclass
class PACIndexConfig:
    """Top-level configuration for the PAC-Index framework."""

    project_name: str = "pac_index"
    version: str = "1.0.0"
    data_dir: str = "benchmarks/data"
    results_dir: str = "results"
    figures_dir: str = "results/figures"
    tables_dir: str = "results/tables"
    statistics_dir: str = "results/statistics"
    log_level: str = "INFO"
    datasets: list[str] = field(default_factory=lambda: ["amzn", "face", "osm", "wiki"])
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # Dataset distribution properties (from paper)
    DATASET_PROPERTIES: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "amzn": {"cv": 0.31, "skewness": 1.2, "gap_rho": 0.08, "kurtosis": 2.1},
        "face": {"cv": 0.72, "skewness": 0.8, "gap_rho": 0.12, "kurtosis": 1.5},
        "osm": {"cv": 1.85, "skewness": 3.4, "gap_rho": 0.31, "kurtosis": 8.7},
        "wiki": {"cv": 0.44, "skewness": 0.5, "gap_rho": 0.15, "kurtosis": 1.8},
    })

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PACIndexConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        config = cls()
        if "data" in raw:
            config.data_dir = raw["data"].get("sosd_dir", config.data_dir)
            config.datasets = raw["data"].get("datasets", config.datasets)
        if "experiment" in raw:
            exp = raw["experiment"]
            config.experiment = ExperimentConfig(
                seeds=exp.get("seeds", config.experiment.seeds),
                num_runs=exp.get("num_runs", config.experiment.num_runs),
                warmup_runs=exp.get("warmup_runs", config.experiment.warmup_runs),
                sample_sizes=exp.get("sample_sizes", config.experiment.sample_sizes),
            )
        if "output" in raw:
            config.results_dir = raw["output"].get("results_dir", config.results_dir)
            config.figures_dir = raw["output"].get("figures_dir", config.figures_dir)
            config.tables_dir = raw["output"].get("tables_dir", config.tables_dir)
            config.statistics_dir = raw["output"].get("statistics_dir", config.statistics_dir)
        if "logging" in raw:
            config.log_level = raw["logging"].get("level", config.log_level)
        return config

    def get_dataset_cv(self, dataset_id: str) -> float:
        """Return the coefficient of variation for a dataset."""
        return self.DATASET_PROPERTIES.get(dataset_id, {}).get("cv", 1.0)

    def get_dataset_gap_rho(self, dataset_id: str) -> float:
        """Return the gap autocorrelation for a dataset."""
        return self.DATASET_PROPERTIES.get(dataset_id, {}).get("gap_rho", 0.0)

    def ensure_dirs(self) -> None:
        """Create output directories if they do not exist."""
        for d in [self.results_dir, self.figures_dir, self.tables_dir, self.statistics_dir]:
            os.makedirs(d, exist_ok=True)
