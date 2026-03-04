"""Tests for workload generation and data loading."""

from __future__ import annotations

import numpy as np
import pytest

from pac_index.workload.generator import WorkloadGenerator
from pac_index.workload.loader import subsample_keys


class TestWorkloadGenerator:

    @pytest.mark.unit
    def test_point_queries_count(self, synthetic_keys: np.ndarray) -> None:
        gen = WorkloadGenerator(seed=42)
        queries = gen.generate_point_queries(synthetic_keys, 1000)
        assert len(queries) == 1000

    @pytest.mark.unit
    def test_point_queries_in_range(self, synthetic_keys: np.ndarray) -> None:
        gen = WorkloadGenerator(seed=42)
        queries = gen.generate_point_queries(synthetic_keys, 100)
        assert np.all(np.isin(queries, synthetic_keys))

    @pytest.mark.unit
    def test_range_queries(self, synthetic_keys: np.ndarray) -> None:
        gen = WorkloadGenerator(seed=42)
        ranges = gen.generate_range_queries(synthetic_keys, 100, range_size=10)
        assert len(ranges) == 100
        for start, end in ranges:
            assert start <= end

    @pytest.mark.unit
    def test_mixed_workload(self, synthetic_keys: np.ndarray) -> None:
        gen = WorkloadGenerator(seed=42)
        mixed = gen.generate_mixed_workload(synthetic_keys, 1000, point_fraction=0.7)
        assert "point_queries" in mixed
        assert "range_queries" in mixed

    @pytest.mark.unit
    def test_deterministic(self, synthetic_keys: np.ndarray) -> None:
        gen1 = WorkloadGenerator(seed=42)
        gen2 = WorkloadGenerator(seed=42)
        q1 = gen1.generate_point_queries(synthetic_keys, 100)
        q2 = gen2.generate_point_queries(synthetic_keys, 100)
        np.testing.assert_array_equal(q1, q2)


class TestSubsample:

    @pytest.mark.unit
    def test_subsample_size(self, synthetic_keys: np.ndarray) -> None:
        sub = subsample_keys(synthetic_keys, 100, seed=42)
        assert len(sub) == 100

    @pytest.mark.unit
    def test_subsample_sorted(self, synthetic_keys: np.ndarray) -> None:
        sub = subsample_keys(synthetic_keys, 100, seed=42)
        assert np.all(sub[:-1] <= sub[1:])

    @pytest.mark.unit
    def test_subsample_larger_than_data(self, synthetic_keys: np.ndarray) -> None:
        sub = subsample_keys(synthetic_keys, len(synthetic_keys) + 100, seed=42)
        np.testing.assert_array_equal(sub, synthetic_keys)
