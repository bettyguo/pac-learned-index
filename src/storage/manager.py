"""Storage manager for loading and managing SOSD benchmark datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages loading, sampling, and caching of SOSD benchmark datasets."""

    DATASET_SIZES = {
        "amzn": 200_000_000,
        "face": 200_000_000,
        "osm": 200_000_000,
        "wiki": 200_000_000,
    }

    def __init__(self, data_dir: str = "benchmarks/data") -> None:
        self.data_dir = Path(data_dir)
        self._cache: dict[str, np.ndarray] = {}

    def load_dataset(self, dataset_id: str, max_keys: int | None = None) -> np.ndarray:
        """Load a sorted dataset from binary file.

        SOSD datasets are stored as sorted arrays of uint64 keys.
        """
        cache_key = f"{dataset_id}_{max_keys}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        filepath = self.data_dir / f"{dataset_id}_200M_uint64"
        if not filepath.exists():
            raise FileNotFoundError(
                f"Dataset not found: {filepath}. "
                f"Run 'make benchmarks' to download SOSD data."
            )

        logger.info("Loading dataset: %s", dataset_id)
        keys = np.fromfile(str(filepath), dtype=np.uint64)
        if max_keys is not None:
            keys = keys[:max_keys]
        keys = np.sort(keys)

        self._cache[cache_key] = keys
        logger.info("Loaded %d keys from %s", len(keys), dataset_id)
        return keys

    def sample_keys(
        self,
        dataset_id: str,
        sample_size: int,
        seed: int = 42,
    ) -> np.ndarray:
        """Sample and sort keys from a dataset."""
        keys = self.load_dataset(dataset_id)
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(keys), size=min(sample_size, len(keys)), replace=False)
        return np.sort(keys[indices])

    def compute_gaps(self, keys: np.ndarray) -> np.ndarray:
        """Compute consecutive key gaps for distribution analysis."""
        return np.diff(keys.astype(np.float64))

    def estimate_cv(self, keys: np.ndarray) -> float:
        """Estimate coefficient of variation from key gaps."""
        gaps = self.compute_gaps(keys)
        if len(gaps) == 0:
            return 0.0
        mean_gap = np.mean(gaps)
        if mean_gap == 0:
            return 0.0
        return float(np.std(gaps) / mean_gap)

    def estimate_gap_autocorrelation(self, keys: np.ndarray, max_lag: int = 1) -> float:
        """Estimate lag-1 autocorrelation of key gaps."""
        gaps = self.compute_gaps(keys)
        if len(gaps) < 2:
            return 0.0
        mean_gap = np.mean(gaps)
        var_gap = np.var(gaps)
        if var_gap == 0:
            return 0.0
        autocov = np.mean((gaps[:-max_lag] - mean_gap) * (gaps[max_lag:] - mean_gap))
        return float(autocov / var_gap)

    def get_dataset_properties(self, keys: np.ndarray) -> dict[str, float]:
        """Compute comprehensive distribution properties for a key array."""
        from scipy import stats as sp_stats

        gaps = self.compute_gaps(keys)
        if len(gaps) == 0:
            return {"cv": 0.0, "skewness": 0.0, "kurtosis": 0.0, "gap_rho": 0.0}

        mean_gap = float(np.mean(gaps))
        std_gap = float(np.std(gaps))
        cv = std_gap / mean_gap if mean_gap > 0 else 0.0

        return {
            "cv": cv,
            "skewness": float(sp_stats.skew(gaps)),
            "kurtosis": float(sp_stats.kurtosis(gaps)),
            "gap_rho": self.estimate_gap_autocorrelation(keys),
        }

    def clear_cache(self) -> None:
        """Clear the dataset cache."""
        self._cache.clear()
