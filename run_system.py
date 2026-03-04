#!/usr/bin/env python3
"""Main entry point for PAC-Index analysis system."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from pac_index.core.config import PACIndexConfig
from pac_index.core.engine import PACIndexEngine
from pac_index.utils.reproducibility import set_seed


@click.command()
@click.option("--config", default="configs/default.yaml", help="Configuration YAML file.")
@click.option("--dataset", default=None, help="Specific dataset to analyze (overrides config).")
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility.")
@click.option("--verbose", is_flag=True, help="Enable verbose output.")
def main(config: str, dataset: str | None, seed: int, verbose: bool) -> None:
    """PAC-Index: PAC learning theory framework for learned database indexes.

    Computes VC dimension bounds, sample complexity, and theory-guided
    hyperparameter selection for learned index architectures.
    """
    set_seed(seed)
    cfg = PACIndexConfig.from_yaml(config)
    if verbose:
        cfg.log_level = "DEBUG"
    if dataset:
        cfg.datasets = [dataset]

    engine = PACIndexEngine(cfg)
    results = engine.run()

    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(title="PAC-Index Theory-Guided Analysis")
    table.add_column("Dataset", style="cyan")
    table.add_column("CV", justify="right")
    table.add_column("Gap rho", justify="right")
    table.add_column("k* (segments)", justify="right")
    table.add_column("VC Dimension", justify="right")
    table.add_column("m_free", justify="right")
    table.add_column("m_dep", justify="right")

    for analysis in results["analyses"]:
        table.add_row(
            analysis["dataset"],
            f"{analysis['cv']:.2f}",
            f"{analysis['gap_rho']:.2f}",
            f"{analysis['k_star']:,}",
            f"{analysis['vc_dimension']:,.0f}",
            f"{analysis['sample_complexity_free']:,}",
            f"{analysis['sample_complexity_dependent']:,}",
        )
    console.print(table)
    console.print(f"[green]Analysis completed in {results['elapsed_seconds']:.2f}s[/green]")


if __name__ == "__main__":
    main()
