#!/usr/bin/env python3
"""Interactive demonstration of PAC-Index framework capabilities."""

from __future__ import annotations

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pac_index.core.engine import PACIndexEngine
from pac_index.core.config import PACIndexConfig
from pac_index.storage.structures import PWLIndex


def demo_vc_bounds() -> None:
    """Demonstrate VC dimension computation for various architectures."""
    console = Console()
    console.print(Panel("[bold]Demo 1: VC Dimension Bounds[/bold]"))

    table = Table(title="VC Dimension Bounds for Learned Index Architectures")
    table.add_column("Architecture", style="cyan")
    table.add_column("Parameters", style="white")
    table.add_column("VC Lower", justify="right")
    table.add_column("VC Upper", justify="right")
    table.add_column("Tight?", justify="center")

    vc_pwl = PACIndexEngine.vc_dim_pwl(100)
    table.add_row("PWL (k=100)", "k=100", f"{vc_pwl.lower_bound:.0f}", f"{vc_pwl.upper_bound:.0f}", "Yes")

    vc_rmi = PACIndexEngine.vc_dim_rmi(2, 100)
    table.add_row("RMI (d=2,w=100)", f"d=2,w=100", f"{vc_rmi.lower_bound:.0f}", f"{vc_rmi.upper_bound:.0f}", "To log")

    vc_alex = PACIndexEngine.vc_dim_alex(10_000_000, 64)
    table.add_row("ALEX (n=10M)", "n=10M,eps=64", f"{vc_alex.lower_bound:.0f}", f"{vc_alex.upper_bound:.0f}", "Yes")

    vc_lipp = PACIndexEngine.vc_dim_lipp(10_000_000)
    table.add_row("LIPP (n=10M)", "n=10M", f"{vc_lipp.lower_bound:.0f}", f"{vc_lipp.upper_bound:.0f}", "Yes")

    vc_rs = PACIndexEngine.vc_dim_radixspline(18, 1000)
    table.add_row("RadixSpline", "r=18,s=1000", f"{vc_rs.lower_bound:.0f}", f"{vc_rs.upper_bound:.0f}", "Yes")

    console.print(table)


def demo_sample_complexity() -> None:
    """Demonstrate sample complexity computation."""
    console = Console()
    console.print(Panel("[bold]Demo 2: Sample Complexity Bounds[/bold]"))

    table = Table(title="Sample Complexity (epsilon=0.01, delta=0.05)")
    table.add_column("Architecture", style="cyan")
    table.add_column("VC Dim", justify="right")
    table.add_column("m (free)", justify="right")
    table.add_column("m (CV=0.31)", justify="right")
    table.add_column("m (CV=1.85)", justify="right")

    for name, vc in [("PWL-100", 280.0), ("PWL-1000", 2800.0), ("RMI-2-100", 20000.0)]:
        m_free = PACIndexEngine.sample_complexity(vc, 0.01, 0.05)
        m_low_cv = PACIndexEngine.sample_complexity_distribution_dependent(vc, 0.01, 0.05, 0.31)
        m_high_cv = PACIndexEngine.sample_complexity_distribution_dependent(vc, 0.01, 0.05, 1.85)
        table.add_row(name, f"{vc:,.0f}", f"{m_free:,}", f"{m_low_cv:,}", f"{m_high_cv:,}")

    console.print(table)


def demo_theory_guided_selection() -> None:
    """Demonstrate theory-guided hyperparameter selection."""
    console = Console()
    console.print(Panel("[bold]Demo 3: Theory-Guided Hyperparameter Selection (Zero Builds)[/bold]"))

    config = PACIndexConfig()
    engine = PACIndexEngine(config)

    table = Table(title="Optimal Segments for Target Error=100, n=200M")
    table.add_column("Dataset", style="cyan")
    table.add_column("CV", justify="right")
    table.add_column("k* (theory)", justify="right")
    table.add_column("k* (empirical)", justify="right")
    table.add_column("Deviation", justify="right")

    empirical_k = {"amzn": 1856, "face": 10789, "osm": 71234, "wiki": 15200}
    for ds in ["amzn", "face", "osm", "wiki"]:
        result = engine.theory_guided_selection(ds, 200_000_000, 100)
        k_emp = empirical_k[ds]
        dev = abs(result["k_star"] - k_emp) / k_emp * 100
        table.add_row(ds, f"{result['cv']:.2f}", f"{result['k_star']:,}", f"{k_emp:,}", f"{dev:.1f}%")

    console.print(table)


def demo_pwl_build() -> None:
    """Demonstrate PWL index construction on synthetic data."""
    console = Console()
    console.print(Panel("[bold]Demo 4: PWL Index Construction[/bold]"))

    np.random.seed(42)
    n = 100_000
    keys = np.sort(np.random.uniform(0, 1e9, n).astype(np.uint64))

    for eps in [16, 32, 64, 128, 256]:
        index = PWLIndex.build_optimal(keys, eps)
        positions = np.arange(n, dtype=np.float64)
        max_err = index.max_error(keys, positions)
        avg_err = index.avg_error(keys, positions)
        console.print(
            f"  eps={eps:>4d}: segments={index.num_segments:>5d}, "
            f"max_error={max_err:>8.1f}, avg_error={avg_err:>8.1f}"
        )


def main() -> None:
    """Run all PAC-Index demos."""
    console = Console()
    console.print("[bold magenta]PAC-Index: Interactive Demonstration[/bold magenta]")
    console.print("=" * 60)

    demo_vc_bounds()
    console.print()
    demo_sample_complexity()
    console.print()
    demo_theory_guided_selection()
    console.print()
    demo_pwl_build()

    console.print()
    console.print("[bold green]All demos completed successfully.[/bold green]")


if __name__ == "__main__":
    main()
