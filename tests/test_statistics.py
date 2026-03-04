"""Tests for statistical analysis utilities."""

from __future__ import annotations

import numpy as np
import pytest

from pac_index.evaluation.statistics import (
    compute_confidence_interval,
    paired_t_test,
    bootstrap_ci,
    pearson_correlation,
    convergence_rate_fit,
    compute_distribution_validation,
)


class TestConfidenceInterval:

    @pytest.mark.unit
    def test_basic_ci(self) -> None:
        data = np.array([10.0, 11.0, 12.0, 9.0, 10.5])
        mean, lower, upper = compute_confidence_interval(data, 0.95)
        assert lower < mean < upper
        assert abs(mean - np.mean(data)) < 1e-10

    @pytest.mark.unit
    def test_single_value(self) -> None:
        data = np.array([5.0])
        mean, lower, upper = compute_confidence_interval(data)
        assert mean == lower == upper == 5.0


class TestPairedTTest:

    @pytest.mark.unit
    def test_identical_no_significance(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = paired_t_test(a, a)
        assert result["p_value"] >= 0.05 or np.isnan(result["p_value"])

    @pytest.mark.unit
    def test_different_significant(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        result = paired_t_test(a, b)
        assert result["p_value"] < 0.05


class TestBootstrapCI:

    @pytest.mark.unit
    def test_bootstrap_bounds(self) -> None:
        data = np.random.default_rng(42).normal(100, 10, 50)
        mean, lower, upper = bootstrap_ci(data, n_bootstrap=5000)
        assert lower < mean < upper


class TestPearsonCorrelation:

    @pytest.mark.unit
    def test_perfect_correlation(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2 * x + 1
        result = pearson_correlation(x, y)
        assert abs(result["r"] - 1.0) < 1e-10

    @pytest.mark.unit
    def test_r_squared(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 5.0, 4.5, 5.5])
        result = pearson_correlation(x, y)
        assert abs(result["r_squared"] - result["r"] ** 2) < 1e-10


class TestConvergenceRateFit:

    @pytest.mark.unit
    def test_sqrt_convergence(self) -> None:
        """Fit error = alpha / sqrt(m) should recover slope ~ -0.5."""
        m = np.array([1e3, 1e4, 1e5, 1e6, 1e7])
        alpha = 10.0
        errors = alpha / np.sqrt(m) + np.random.default_rng(42).normal(0, 0.01, len(m))
        errors = np.maximum(errors, 1e-10)
        result = convergence_rate_fit(m, errors)
        assert abs(result["slope"] - (-0.5)) < 0.1


class TestDistributionValidation:

    @pytest.mark.unit
    def test_amzn_prediction(self) -> None:
        result = compute_distribution_validation(0.31, 0.08)
        assert result["cv"] == 0.31
        assert result["gap_rho"] == 0.08
        assert result["predicted_coeff"] > 0

    @pytest.mark.unit
    def test_cv_ordering(self) -> None:
        low = compute_distribution_validation(0.31, 0.0)
        high = compute_distribution_validation(1.85, 0.0)
        assert high["predicted_coeff"] > low["predicted_coeff"]
