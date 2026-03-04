#!/usr/bin/env python3
"""Experiment runner for PAC-Index validation experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from pac_index.core.config import PACIndexConfig
from pac_index.core.pipeline import ExperimentPipeline
from pac_index.utils.reproducibility import set_seed


@click.command()
@click.option("--config", default="configs/default.yaml", help="Configuration YAML file.")
@click.option("--experiment", default="full",
              type=click.Choice(["full", "vc", "sample_complexity", "hpo", "workflow"]),
              help="Experiment to run.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--output", default=None, help="Output JSON file path.")
def main(config: str, experiment: str, seed: int, output: str | None) -> None:
    """Run PAC-Index validation experiments.

    Experiments include VC dimension validation, sample complexity validation,
    HPO method comparison, and practical workflow demonstration.
    """
    set_seed(seed)
    cfg = PACIndexConfig.from_yaml(config)
    pipeline = ExperimentPipeline(cfg)

    from rich.console import Console
    console = Console()

    experiment_map = {
        "full": pipeline.run_full_pipeline,
        "vc": pipeline.run_vc_dimension_validation,
        "sample_complexity": pipeline.run_sample_complexity_validation,
        "hpo": pipeline.run_hpo_comparison,
        "workflow": pipeline.run_practical_workflow,
    }

    console.print(f"[bold cyan]Running experiment: {experiment}[/bold cyan]")
    results = experiment_map[experiment]()

    output_path = output or str(Path(cfg.results_dir) / f"{experiment}_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    console.print(f"[green]Results saved to {output_path}[/green]")


if __name__ == "__main__":
    main()
