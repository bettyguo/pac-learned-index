"""Data loader utilities for SOSD benchmark datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SOSD_DATASETS = {
    "amzn": "books_200M_uint64",
    "face": "fb_200M_uint64",
    "osm": "osm_cellids_200M_uint64",
    "wiki": "wiki_ts_200M_uint64",
}

SOSD_URLS = {
    "amzn": "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/MZZUP2",
    "face": "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/GKMPEH",
    "osm": "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/5M0X0Z",
    "wiki": "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/JGVF9A/SVN8PI",
}


def load_sosd_dataset(
    dataset_id: str,
    data_dir: str = "benchmarks/data",
    max_keys: int | None = None,
) -> np.ndarray:
    """Load a SOSD benchmark dataset from binary file.

    Binary format: sorted array of uint64 values (little-endian).
    """
    filename = SOSD_DATASETS.get(dataset_id, f"{dataset_id}_200M_uint64")
    filepath = Path(data_dir) / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            f"Run 'make benchmarks' or 'bash scripts/setup_benchmarks.sh' to download."
        )

    logger.info("Loading %s from %s", dataset_id, filepath)
    keys = np.fromfile(str(filepath), dtype=np.uint64)
    if max_keys is not None and max_keys < len(keys):
        keys = keys[:max_keys]
    return np.sort(keys)


def subsample_keys(
    keys: np.ndarray,
    sample_size: int,
    seed: int = 42,
) -> np.ndarray:
    """Subsample and sort keys from a full dataset."""
    if sample_size >= len(keys):
        return keys
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(keys), size=sample_size, replace=False)
    return np.sort(keys[indices])
