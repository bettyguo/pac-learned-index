"""Theory-guided query optimization using PAC-Index bounds."""

from __future__ import annotations

import math
from typing import Any

from pac_index.core.engine import PACIndexEngine


def recommend_error_bound(
    n: int,
    cv: float,
    target_latency_ns: int = 200,
    gap_rho: float = 0.0,
) -> dict[str, Any]:
    """Recommend an error bound based on theoretical analysis.

    Uses the relationship between prediction error and query latency
    to suggest an appropriate error bound for a given latency target.
    """
    epsilon_candidates = [8, 16, 32, 64, 128, 256, 512]
    best: dict[str, Any] = {}

    for eps in epsilon_candidates:
        k_star = PACIndexEngine.optimal_segments(n, cv, eps)
        vc = PACIndexEngine.vc_dim_pwl(k_star)
        m_needed = PACIndexEngine.sample_complexity_distribution_dependent(
            vc.vc_dimension, eps / n, 0.05, cv, gap_rho
        )
        best[eps] = {
            "epsilon": eps,
            "k_star": k_star,
            "vc_dimension": vc.vc_dimension,
            "sample_complexity": m_needed,
            "feasible": m_needed <= n,
        }

    return best


def architecture_recommendation(
    n: int,
    cv: float,
    update_rate: float = 0.0,
) -> str:
    """Recommend a learned index architecture based on dataset properties.

    Based on VC dimension hierarchy and update requirements.
    """
    if update_rate > 0.1:
        if n < 10_000_000:
            return "ALEX"
        return "LIPP"

    if cv < 0.5:
        return "PGM-index"
    elif cv < 1.0:
        return "RadixSpline"
    else:
        return "RMI"
