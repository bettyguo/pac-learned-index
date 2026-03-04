"""Main PAC-Index engine: orchestrates analysis, experiments, and evaluation."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np

from pac_index.core.config import PACIndexConfig
from pac_index.utils.reproducibility import set_seed, get_system_info

logger = logging.getLogger(__name__)


@dataclass
class VCDimensionResult:
    """Result of VC dimension computation for a learned index class."""

    architecture: str
    parameters: dict[str, Any]
    vc_dimension: float
    lower_bound: float
    upper_bound: float


@dataclass
class SampleComplexityResult:
    """Result of sample complexity computation."""

    architecture: str
    vc_dimension: float
    epsilon: float
    delta: float
    sample_complexity: int
    distribution_dependent: bool = False
    cv: float | None = None


class PACIndexEngine:
    """Core engine for PAC-Index theoretical analysis and experimental validation."""

    def __init__(self, config: PACIndexConfig) -> None:
        self.config = config
        self.config.ensure_dirs()
        logging.basicConfig(level=getattr(logging, config.log_level))

    # ─────────────────────────────────────────────────────────────────────
    # VC Dimension Bounds
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def vc_dim_pwl(k: int) -> VCDimensionResult:
        """Compute VC dimension bounds for piecewise linear model with k segments.

        Theorem: 2k <= VC(H_PWL^k) <= 3k - 1.
        """
        lower = 2 * k
        upper = 3 * k - 1
        estimated = 2.8 * k
        return VCDimensionResult(
            architecture="PWL",
            parameters={"k": k},
            vc_dimension=estimated,
            lower_bound=lower,
            upper_bound=upper,
        )

    @staticmethod
    def vc_dim_rmi(d: int, w: int) -> VCDimensionResult:
        """Compute VC dimension bounds for RMI with d levels and branching factor w.

        Theorem: VC(H_RMI^{d,w}) = O(d * w^d * log(w * d)).
        """
        total_params = 2 * (w**d - 1) // (w - 1) if w > 1 else 2 * d
        vc_upper = d * (w**d) * math.log2(w * d) if w > 1 and d > 0 else 2
        return VCDimensionResult(
            architecture="RMI",
            parameters={"d": d, "w": w, "total_params": total_params},
            vc_dimension=vc_upper,
            lower_bound=total_params,
            upper_bound=vc_upper,
        )

    @staticmethod
    def vc_dim_alex(n: int, epsilon: float) -> VCDimensionResult:
        """Compute VC dimension bounds for ALEX.

        Theorem: VC(ALEX) = O(n / epsilon).
        """
        vc = n / epsilon
        return VCDimensionResult(
            architecture="ALEX",
            parameters={"n": n, "epsilon": epsilon},
            vc_dimension=vc,
            lower_bound=vc * 0.5,
            upper_bound=vc,
        )

    @staticmethod
    def vc_dim_lipp(n: int) -> VCDimensionResult:
        """Compute VC dimension bounds for LIPP.

        Theorem: VC(LIPP) = O(n).
        """
        return VCDimensionResult(
            architecture="LIPP",
            parameters={"n": n},
            vc_dimension=float(n),
            lower_bound=float(n) * 0.5,
            upper_bound=float(n),
        )

    @staticmethod
    def vc_dim_radixspline(r: int, s: int) -> VCDimensionResult:
        """Compute VC dimension bounds for RadixSpline.

        Theorem: VC(RadixSpline) = O(2^r * s).
        """
        vc = (2**r) * s
        return VCDimensionResult(
            architecture="RadixSpline",
            parameters={"r": r, "s": s},
            vc_dimension=float(vc),
            lower_bound=float(vc) * 0.5,
            upper_bound=float(vc),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Sample Complexity Bounds
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def sample_complexity(
        vc_dim: float,
        epsilon: float,
        delta: float = 0.05,
    ) -> int:
        """Compute distribution-free sample complexity.

        m = (2 / epsilon^2) * (d_VC * log(2e / epsilon) + log(2 / delta))
        """
        m = (2.0 / epsilon**2) * (
            vc_dim * math.log(2 * math.e / epsilon) + math.log(2.0 / delta)
        )
        return int(math.ceil(m))

    @staticmethod
    def sample_complexity_distribution_dependent(
        vc_dim: float,
        epsilon: float,
        delta: float,
        cv: float,
        gap_rho: float = 0.0,
    ) -> int:
        """Compute distribution-dependent sample complexity with CV refinement.

        m_eff = O((d_VC * CV^2) / ((1 - rho) * epsilon^2) + log(1/delta) / epsilon^2)
        """
        correlation_factor = 1.0 / (1.0 - gap_rho) if gap_rho < 1.0 else float("inf")
        m = (vc_dim * cv**2 * correlation_factor) / epsilon**2 + math.log(1.0 / delta) / epsilon**2
        return int(math.ceil(m))

    # ─────────────────────────────────────────────────────────────────────
    # Theory-Guided Hyperparameter Selection
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def optimal_segments(n: int, cv: float, epsilon: float) -> int:
        """Compute optimal number of segments for PWL index.

        k* = n * CV^2 / epsilon^2 (Equation from paper).
        """
        k_star = n * cv**2 / epsilon**2
        return max(1, int(math.ceil(k_star)))

    def theory_guided_selection(
        self, dataset_id: str, n: int, target_epsilon: float
    ) -> dict[str, Any]:
        """Perform theory-guided hyperparameter selection (zero index builds).

        Returns the recommended number of segments and estimated metrics.
        """
        cv = self.config.get_dataset_cv(dataset_id)
        gap_rho = self.config.get_dataset_gap_rho(dataset_id)
        k_star = self.optimal_segments(n, cv, target_epsilon)
        vc_result = self.vc_dim_pwl(k_star)
        m_free = self.sample_complexity(vc_result.vc_dimension, target_epsilon / n)
        m_dep = self.sample_complexity_distribution_dependent(
            vc_result.vc_dimension, target_epsilon / n, 0.05, cv, gap_rho
        )
        return {
            "dataset": dataset_id,
            "n": n,
            "target_epsilon": target_epsilon,
            "cv": cv,
            "gap_rho": gap_rho,
            "k_star": k_star,
            "vc_dimension": vc_result.vc_dimension,
            "sample_complexity_free": m_free,
            "sample_complexity_dependent": m_dep,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Convergence Rate Prediction
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def predicted_convergence_rate(cv: float) -> float:
        """Predict convergence coefficient: error ~ coeff / sqrt(m).

        From distribution-dependent analysis, coeff proportional to CV^2.
        """
        return cv**2

    # ─────────────────────────────────────────────────────────────────────
    # CLI Entry Point
    # ─────────────────────────────────────────────────────────────────────

    def run(self, experiment_name: str = "default") -> dict[str, Any]:
        """Run a complete PAC-Index analysis pipeline."""
        logger.info("Starting PAC-Index analysis: %s", experiment_name)
        start = time.time()
        results: dict[str, Any] = {"experiment": experiment_name, "analyses": []}

        for ds in self.config.datasets:
            cv = self.config.get_dataset_cv(ds)
            analysis = self.theory_guided_selection(ds, 200_000_000, 100)
            results["analyses"].append(analysis)
            logger.info("Dataset %s: k*=%d, CV=%.2f", ds, analysis["k_star"], cv)

        results["elapsed_seconds"] = time.time() - start
        logger.info("Analysis complete in %.2f seconds", results["elapsed_seconds"])
        return results


@click.command()
@click.option("--config", default="configs/default.yaml", help="Path to configuration YAML file.")
@click.option("--experiment", default="default", help="Experiment name to run.")
@click.option("--seed", default=42, type=int, help="Random seed.")
def main(config: str, experiment: str, seed: int) -> None:
    """PAC-Index: PAC learning theory framework for learned database indexes."""
    set_seed(seed)
    cfg = PACIndexConfig.from_yaml(config)
    engine = PACIndexEngine(cfg)
    results = engine.run(experiment)

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="PAC-Index Theory-Guided Analysis")
    table.add_column("Dataset", style="cyan")
    table.add_column("CV", justify="right")
    table.add_column("k*", justify="right")
    table.add_column("VC Dim", justify="right")
    for analysis in results["analyses"]:
        table.add_row(
            analysis["dataset"],
            f"{analysis['cv']:.2f}",
            f"{analysis['k_star']:,}",
            f"{analysis['vc_dimension']:,.0f}",
        )
    console.print(table)


if __name__ == "__main__":
    main()
