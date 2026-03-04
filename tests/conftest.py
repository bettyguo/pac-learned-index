"""Shared test fixtures and configuration."""

from __future__ import annotations

import numpy as np
import pytest

from pac_index.core.config import PACIndexConfig
from pac_index.core.engine import PACIndexEngine
from pac_index.storage.structures import PWLIndex


@pytest.fixture
def default_config() -> PACIndexConfig:
    """Return default PAC-Index configuration."""
    return PACIndexConfig()


@pytest.fixture
def engine(default_config: PACIndexConfig) -> PACIndexEngine:
    """Return a PACIndexEngine with default config."""
    return PACIndexEngine(default_config)


@pytest.fixture
def synthetic_keys() -> np.ndarray:
    """Generate sorted synthetic keys for testing."""
    rng = np.random.default_rng(42)
    keys = np.sort(rng.integers(0, 10**9, size=10000, dtype=np.uint64))
    return keys


@pytest.fixture
def synthetic_positions(synthetic_keys: np.ndarray) -> np.ndarray:
    """Generate positions matching synthetic keys."""
    return np.arange(len(synthetic_keys), dtype=np.float64)


@pytest.fixture
def small_pwl_index(synthetic_keys: np.ndarray) -> PWLIndex:
    """Build a small PWL index for testing."""
    return PWLIndex.build_optimal(synthetic_keys, error_bound=64)


@pytest.fixture
def sample_sizes() -> list[int]:
    """Standard sample sizes for testing."""
    return [1000, 5000, 10000]


@pytest.fixture
def dataset_cvs() -> dict[str, float]:
    """CV values for all datasets (from paper)."""
    return {"amzn": 0.31, "face": 0.72, "osm": 1.85, "wiki": 0.44}


@pytest.fixture
def dataset_gap_rhos() -> dict[str, float]:
    """Gap autocorrelation values for all datasets (from paper)."""
    return {"amzn": 0.08, "face": 0.12, "osm": 0.31, "wiki": 0.15}
