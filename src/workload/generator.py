"""Workload generation for benchmark evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


class WorkloadGenerator:
    """Generate query workloads for index benchmarking."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_point_queries(
        self,
        keys: np.ndarray,
        num_queries: int,
        distribution: str = "uniform",
    ) -> np.ndarray:
        """Generate point query workload from the key space.

        Args:
            keys: Sorted array of keys.
            num_queries: Number of queries to generate.
            distribution: Query key distribution ('uniform' or 'zipfian').
        """
        n = len(keys)
        if distribution == "uniform":
            indices = self.rng.integers(0, n, size=num_queries)
        elif distribution == "zipfian":
            alpha = 0.99
            probs = np.array([1.0 / (i + 1) ** alpha for i in range(n)])
            probs /= probs.sum()
            indices = self.rng.choice(n, size=num_queries, p=probs)
        else:
            indices = self.rng.integers(0, n, size=num_queries)
        return keys[indices]

    def generate_range_queries(
        self,
        keys: np.ndarray,
        num_queries: int,
        range_size: int = 100,
    ) -> list[tuple[int, int]]:
        """Generate range query workload.

        Returns list of (start_key, end_key) tuples.
        """
        n = len(keys)
        start_indices = self.rng.integers(0, max(1, n - range_size), size=num_queries)
        return [
            (int(keys[i]), int(keys[min(i + range_size, n - 1)]))
            for i in start_indices
        ]

    def generate_mixed_workload(
        self,
        keys: np.ndarray,
        num_queries: int,
        point_fraction: float = 0.7,
    ) -> dict[str, Any]:
        """Generate a mixed workload of point and range queries."""
        num_point = int(num_queries * point_fraction)
        num_range = num_queries - num_point
        return {
            "point_queries": self.generate_point_queries(keys, num_point),
            "range_queries": self.generate_range_queries(keys, num_range),
        }
