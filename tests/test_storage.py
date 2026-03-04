"""Tests for storage module: PWL index, RMI, dataset manager."""

from __future__ import annotations

import numpy as np
import pytest

from pac_index.storage.structures import PWLIndex, PWLSegment, RMIIndex
from pac_index.storage.manager import DatasetManager


class TestPWLIndex:
    """Test piecewise linear index construction and queries."""

    @pytest.mark.unit
    def test_build_creates_segments(self, synthetic_keys: np.ndarray) -> None:
        index = PWLIndex.build_optimal(synthetic_keys, error_bound=64)
        assert index.num_segments > 0

    @pytest.mark.unit
    def test_error_within_bound(self, synthetic_keys: np.ndarray) -> None:
        """Maximum error must respect the error bound."""
        error_bound = 128
        index = PWLIndex.build_optimal(synthetic_keys, error_bound=error_bound)
        positions = np.arange(len(synthetic_keys), dtype=np.float64)
        max_err = index.max_error(synthetic_keys, positions)
        assert max_err <= error_bound * 1.01  # Allow 1% floating point tolerance

    @pytest.mark.unit
    def test_more_segments_lower_error(self, synthetic_keys: np.ndarray) -> None:
        """Tighter error bound should produce more segments."""
        index_loose = PWLIndex.build_optimal(synthetic_keys, error_bound=256)
        index_tight = PWLIndex.build_optimal(synthetic_keys, error_bound=16)
        assert index_tight.num_segments >= index_loose.num_segments

    @pytest.mark.unit
    def test_predict_returns_float(self, small_pwl_index: PWLIndex) -> None:
        pred = small_pwl_index.predict(500_000_000.0)
        assert isinstance(pred, float)

    @pytest.mark.unit
    def test_empty_keys(self) -> None:
        index = PWLIndex.build_optimal(np.array([], dtype=np.uint64), error_bound=64)
        assert index.num_segments == 0


class TestPWLSegment:
    """Test individual PWL segment operations."""

    @pytest.mark.unit
    def test_predict(self) -> None:
        seg = PWLSegment(0.0, 100.0, slope=2.0, intercept=10.0)
        assert seg.predict(5.0) == pytest.approx(20.0)

    @pytest.mark.unit
    def test_error(self) -> None:
        seg = PWLSegment(0.0, 100.0, slope=1.0, intercept=0.0)
        assert seg.error(5.0, 10.0) == pytest.approx(5.0)


class TestRMIIndex:
    """Test RMI index properties."""

    @pytest.mark.unit
    def test_total_models(self) -> None:
        rmi = RMIIndex(depth=2, branching_factor=10)
        assert rmi.total_models == 11  # 1 + 10

    @pytest.mark.unit
    def test_total_parameters(self) -> None:
        rmi = RMIIndex(depth=2, branching_factor=10)
        assert rmi.total_parameters == 22  # 2 * 11


class TestDatasetManager:
    """Test dataset manager operations."""

    @pytest.mark.unit
    def test_estimate_cv(self, synthetic_keys: np.ndarray) -> None:
        manager = DatasetManager()
        cv = manager.estimate_cv(synthetic_keys)
        assert cv >= 0.0

    @pytest.mark.unit
    def test_estimate_gap_autocorrelation(self, synthetic_keys: np.ndarray) -> None:
        manager = DatasetManager()
        rho = manager.estimate_gap_autocorrelation(synthetic_keys)
        assert -1.0 <= rho <= 1.0

    @pytest.mark.unit
    def test_compute_gaps(self, synthetic_keys: np.ndarray) -> None:
        manager = DatasetManager()
        gaps = manager.compute_gaps(synthetic_keys)
        assert len(gaps) == len(synthetic_keys) - 1
        assert np.all(gaps >= 0)
