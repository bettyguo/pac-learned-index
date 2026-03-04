"""Data structures for learned index analysis: PWL segments, RMI tree, etc."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PWLSegment:
    """A single segment of a piecewise linear index."""

    breakpoint_left: float
    breakpoint_right: float
    slope: float
    intercept: float

    def predict(self, key: float) -> float:
        """Predict position for a key within this segment."""
        return self.slope * key + self.intercept

    def error(self, key: float, true_pos: float) -> float:
        """Compute absolute prediction error."""
        return abs(self.predict(key) - true_pos)


@dataclass
class PWLIndex:
    """Piecewise linear learned index with k segments."""

    segments: list[PWLSegment] = field(default_factory=list)

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    def predict(self, key: float) -> float:
        """Predict position for a query key via binary search over segments."""
        lo, hi = 0, len(self.segments) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if key < self.segments[mid].breakpoint_right:
                hi = mid
            else:
                lo = mid + 1
        return self.segments[lo].predict(key)

    def max_error(self, keys: np.ndarray, positions: np.ndarray) -> float:
        """Compute maximum prediction error over all keys."""
        errors = np.array([
            abs(self.predict(float(k)) - float(p))
            for k, p in zip(keys, positions)
        ])
        return float(np.max(errors)) if len(errors) > 0 else 0.0

    def avg_error(self, keys: np.ndarray, positions: np.ndarray) -> float:
        """Compute average prediction error over all keys."""
        errors = np.array([
            abs(self.predict(float(k)) - float(p))
            for k, p in zip(keys, positions)
        ])
        return float(np.mean(errors)) if len(errors) > 0 else 0.0

    @classmethod
    def build_optimal(cls, keys: np.ndarray, error_bound: int) -> "PWLIndex":
        """Build an optimal PWL index using a greedy segmentation algorithm.

        This implements a simplified version of the PGM-index construction.
        """
        n = len(keys)
        if n == 0:
            return cls(segments=[])

        positions = np.arange(n, dtype=np.float64)
        segments: list[PWLSegment] = []
        start = 0

        while start < n:
            end = start + 1
            while end < n:
                seg_keys = keys[start:end + 1].astype(np.float64)
                seg_pos = positions[start:end + 1]
                if len(seg_keys) < 2:
                    end += 1
                    continue
                slope, intercept = np.polyfit(seg_keys, seg_pos, 1)
                predictions = slope * seg_keys + intercept
                max_err = np.max(np.abs(predictions - seg_pos))
                if max_err > error_bound:
                    break
                end += 1

            seg_keys = keys[start:end].astype(np.float64)
            seg_pos = positions[start:end]
            if len(seg_keys) >= 2:
                slope, intercept = np.polyfit(seg_keys, seg_pos, 1)
            else:
                slope, intercept = 0.0, float(seg_pos[0])

            segments.append(PWLSegment(
                breakpoint_left=float(keys[start]),
                breakpoint_right=float(keys[end - 1]) if end <= n else float(keys[-1]),
                slope=slope,
                intercept=intercept,
            ))
            start = end

        return cls(segments=segments)


@dataclass
class RMINode:
    """A single node in a Recursive Model Index."""

    level: int
    index: int
    slope: float = 0.0
    intercept: float = 0.0


@dataclass
class RMIIndex:
    """Recursive Model Index with d levels and branching factor w."""

    depth: int
    branching_factor: int
    nodes: dict[tuple[int, int], RMINode] = field(default_factory=dict)

    @property
    def total_models(self) -> int:
        return sum(self.branching_factor ** i for i in range(self.depth))

    @property
    def total_parameters(self) -> int:
        return 2 * self.total_models

    def predict(self, key: float) -> float:
        """Predict position via recursive routing through the model tree."""
        node_idx = 0
        for level in range(self.depth):
            node = self.nodes.get((level, node_idx))
            if node is None:
                return 0.0
            prediction = node.slope * key + node.intercept
            if level < self.depth - 1:
                node_idx = int(prediction * self.branching_factor)
                node_idx = max(0, min(node_idx, self.branching_factor ** (level + 1) - 1))
            else:
                return prediction
        return 0.0
