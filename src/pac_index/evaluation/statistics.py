"""Statistical analysis utilities for PAC-Index experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def compute_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute confidence interval for a set of measurements.

    Returns (mean, lower_bound, upper_bound).
    """
    n = len(data)
    if n < 2:
        m = float(data[0]) if n == 1 else 0.0
        return m, m, m
    mean = float(np.mean(data))
    se = float(stats.sem(data))
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_crit * se
    return mean, mean - margin, mean + margin


def paired_t_test(
    system_a: np.ndarray,
    system_b: np.ndarray,
) -> dict[str, float]:
    """Perform paired t-test between two systems across multiple runs."""
    t_stat, p_value = stats.ttest_rel(system_a, system_b)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.001,
    }


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Returns (mean, lower_bound, upper_bound).
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    means = np.array([
        np.mean(rng.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(means, 100 * alpha))
    upper = float(np.percentile(means, 100 * (1 - alpha)))
    return float(np.mean(data)), lower, upper


def speedup_analysis(
    baseline_times: np.ndarray,
    system_times: np.ndarray,
) -> dict[str, float]:
    """Compute speedup statistics."""
    speedups = baseline_times / system_times
    return {
        "mean_speedup": float(np.mean(speedups)),
        "median_speedup": float(np.median(speedups)),
        "min_speedup": float(np.min(speedups)),
        "max_speedup": float(np.max(speedups)),
    }


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute Pearson correlation coefficient with p-value."""
    r, p = stats.pearsonr(x, y)
    return {
        "r": float(r),
        "p_value": float(p),
        "r_squared": float(r**2),
    }


def convergence_rate_fit(
    sample_sizes: np.ndarray,
    errors: np.ndarray,
) -> dict[str, float]:
    """Fit convergence rate: error = alpha / sqrt(m).

    Returns fitted alpha and goodness of fit.
    """
    log_m = np.log(sample_sizes.astype(float))
    log_err = np.log(errors.astype(float) + 1e-10)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_m, log_err)
    alpha = float(np.exp(intercept))
    return {
        "alpha": alpha,
        "slope": float(slope),
        "r_squared": float(r_value**2),
        "p_value": float(p_value),
    }


def compute_distribution_validation(
    cv: float,
    gap_rho: float,
) -> dict[str, float]:
    """Compute predicted convergence coefficient from distribution properties.

    Predicted coefficient = CV^2 / (1 - rho) (from distribution-dependent bounds).
    """
    correction = 1.0 / (1.0 - gap_rho) if gap_rho < 1.0 else float("inf")
    predicted = cv**2 * correction
    return {
        "cv": cv,
        "gap_rho": gap_rho,
        "predicted_coeff": predicted,
        "correction_factor": correction,
    }
