"""Tests for core engine: VC dimension bounds, sample complexity, HPO."""

from __future__ import annotations

import math

import pytest
import numpy as np

from pac_index.core.config import PACIndexConfig
from pac_index.core.engine import PACIndexEngine


class TestVCDimensionBounds:
    """Test VC dimension computation for all architectures."""

    @pytest.mark.unit
    def test_pwl_vc_bounds(self) -> None:
        """VC(PWL^k) must satisfy 2k <= VC <= 3k-1."""
        for k in [1, 10, 50, 100, 500, 1000]:
            result = PACIndexEngine.vc_dim_pwl(k)
            assert result.lower_bound == 2 * k
            assert result.upper_bound == 3 * k - 1
            assert result.lower_bound <= result.vc_dimension <= result.upper_bound
            assert result.architecture == "PWL"

    @pytest.mark.unit
    def test_pwl_empirical_vc(self) -> None:
        """Empirical VC dimension should be approximately 2.8k."""
        result = PACIndexEngine.vc_dim_pwl(100)
        assert abs(result.vc_dimension - 280.0) < 1e-10

    @pytest.mark.unit
    def test_rmi_vc_bounds(self) -> None:
        """RMI VC dimension should be O(d * w^d * log(wd))."""
        result = PACIndexEngine.vc_dim_rmi(2, 100)
        assert result.vc_dimension > 0
        assert result.architecture == "RMI"
        assert result.parameters["d"] == 2
        assert result.parameters["w"] == 100

    @pytest.mark.unit
    def test_alex_vc_bounds(self) -> None:
        """ALEX VC dimension should be O(n/epsilon)."""
        result = PACIndexEngine.vc_dim_alex(10_000_000, 64)
        expected = 10_000_000 / 64
        assert abs(result.vc_dimension - expected) < 1e-6
        assert result.architecture == "ALEX"

    @pytest.mark.unit
    def test_lipp_vc_bounds(self) -> None:
        """LIPP VC dimension should be O(n)."""
        result = PACIndexEngine.vc_dim_lipp(10_000_000)
        assert result.vc_dimension == 10_000_000.0
        assert result.architecture == "LIPP"

    @pytest.mark.unit
    def test_radixspline_vc_bounds(self) -> None:
        """RadixSpline VC dimension should be O(2^r * s)."""
        result = PACIndexEngine.vc_dim_radixspline(18, 1000)
        expected = (2**18) * 1000
        assert result.vc_dimension == float(expected)
        assert result.architecture == "RadixSpline"


class TestSampleComplexity:
    """Test sample complexity computations."""

    @pytest.mark.unit
    def test_sample_complexity_positive(self) -> None:
        """Sample complexity must be a positive integer."""
        m = PACIndexEngine.sample_complexity(100, 0.01, 0.05)
        assert m > 0
        assert isinstance(m, int)

    @pytest.mark.unit
    def test_sample_complexity_monotone_vc(self) -> None:
        """Higher VC dimension should require more samples."""
        m1 = PACIndexEngine.sample_complexity(100, 0.01, 0.05)
        m2 = PACIndexEngine.sample_complexity(200, 0.01, 0.05)
        assert m2 > m1

    @pytest.mark.unit
    def test_sample_complexity_monotone_epsilon(self) -> None:
        """Smaller epsilon should require more samples."""
        m1 = PACIndexEngine.sample_complexity(100, 0.1, 0.05)
        m2 = PACIndexEngine.sample_complexity(100, 0.01, 0.05)
        assert m2 > m1

    @pytest.mark.unit
    def test_distribution_dependent_tighter_low_cv(self) -> None:
        """Distribution-dependent bound should be tighter for low CV."""
        m_free = PACIndexEngine.sample_complexity(100, 0.01, 0.05)
        m_dep = PACIndexEngine.sample_complexity_distribution_dependent(
            100, 0.01, 0.05, cv=0.31, gap_rho=0.08
        )
        assert m_dep < m_free

    @pytest.mark.unit
    def test_distribution_dependent_cv_ordering(self) -> None:
        """Higher CV should yield larger sample complexity."""
        m_low_cv = PACIndexEngine.sample_complexity_distribution_dependent(
            100, 0.01, 0.05, cv=0.31, gap_rho=0.0
        )
        m_high_cv = PACIndexEngine.sample_complexity_distribution_dependent(
            100, 0.01, 0.05, cv=1.85, gap_rho=0.0
        )
        assert m_high_cv > m_low_cv


class TestTheoryGuidedSelection:
    """Test theory-guided hyperparameter selection."""

    @pytest.mark.unit
    def test_optimal_segments_positive(self) -> None:
        k = PACIndexEngine.optimal_segments(200_000_000, 0.31, 100)
        assert k >= 1

    @pytest.mark.unit
    def test_cv_affects_segments(self) -> None:
        """Higher CV should require more segments for same error."""
        k_low = PACIndexEngine.optimal_segments(200_000_000, 0.31, 100)
        k_high = PACIndexEngine.optimal_segments(200_000_000, 1.85, 100)
        assert k_high > k_low

    @pytest.mark.unit
    def test_theory_guided_returns_dict(self, engine: PACIndexEngine) -> None:
        result = engine.theory_guided_selection("amzn", 200_000_000, 100)
        assert "k_star" in result
        assert "cv" in result
        assert result["cv"] == 0.31
        assert result["k_star"] >= 1


class TestConfig:
    """Test configuration management."""

    @pytest.mark.unit
    def test_default_config(self) -> None:
        config = PACIndexConfig()
        assert config.project_name == "pac_index"
        assert len(config.datasets) == 4
        assert config.experiment.num_runs == 10

    @pytest.mark.unit
    def test_dataset_cv_values(self) -> None:
        """CV values must match paper exactly."""
        config = PACIndexConfig()
        assert config.get_dataset_cv("amzn") == 0.31
        assert config.get_dataset_cv("face") == 0.72
        assert config.get_dataset_cv("osm") == 1.85
        assert config.get_dataset_cv("wiki") == 0.44

    @pytest.mark.unit
    def test_dataset_gap_rho_values(self) -> None:
        """Gap autocorrelation values must match paper exactly."""
        config = PACIndexConfig()
        assert config.get_dataset_gap_rho("amzn") == 0.08
        assert config.get_dataset_gap_rho("face") == 0.12
        assert config.get_dataset_gap_rho("osm") == 0.31
        assert config.get_dataset_gap_rho("wiki") == 0.15
