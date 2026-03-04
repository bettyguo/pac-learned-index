"""Tests for evaluation metrics and paper reference data."""

from __future__ import annotations

import numpy as np
import pytest

from pac_index.evaluation.metrics import (
    PredictionMetrics,
    LatencyMetrics,
    IndexBenchmarkResult,
    LATENCY_TABLE_DATA,
    BASELINE_COMPARISON_DATA,
    DISTRIBUTION_VALIDATION_DATA,
    HPO_COMPARISON_DATA,
    PRACTICAL_WORKFLOW_DATA,
)


class TestPredictionMetrics:

    @pytest.mark.unit
    def test_from_errors(self) -> None:
        errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = PredictionMetrics.from_errors(errors)
        assert metrics.max_error == 5.0
        assert metrics.avg_error == 3.0
        assert metrics.median_error == 3.0

    @pytest.mark.unit
    def test_empty_errors(self) -> None:
        metrics = PredictionMetrics.from_errors(np.array([]))
        assert metrics.max_error == 0.0


class TestLatencyMetrics:

    @pytest.mark.unit
    def test_from_latencies(self) -> None:
        latencies = np.random.default_rng(42).normal(200, 50, 1000)
        metrics = LatencyMetrics.from_latencies(latencies)
        assert metrics.p50 > 0
        assert metrics.p99 >= metrics.p50


class TestPaperReferenceData:
    """Validate that reference data matches paper exactly."""

    @pytest.mark.unit
    def test_latency_datasets(self) -> None:
        assert set(LATENCY_TABLE_DATA.keys()) == {"amzn", "face", "osm", "wiki"}

    @pytest.mark.unit
    def test_latency_sample_sizes(self) -> None:
        for ds in LATENCY_TABLE_DATA:
            assert set(LATENCY_TABLE_DATA[ds].keys()) == {10_000, 100_000, 1_000_000, 10_000_000, 100_000_000}

    @pytest.mark.unit
    def test_amzn_latency_values(self) -> None:
        """Spot check amzn values from paper."""
        amzn = LATENCY_TABLE_DATA["amzn"]
        assert amzn[10_000]["eps_max"] == 1247
        assert amzn[10_000]["p50"] == 892
        assert amzn[10_000_000]["eps_max"] == 47
        assert amzn[10_000_000]["p50"] == 142
        assert amzn[100_000_000]["eps_max"] == 18
        assert amzn[100_000_000]["p50"] == 121

    @pytest.mark.unit
    def test_osm_latency_values(self) -> None:
        """Spot check osm values from paper."""
        osm = LATENCY_TABLE_DATA["osm"]
        assert osm[10_000]["eps_max"] == 3421
        assert osm[100_000_000]["p50"] == 178

    @pytest.mark.unit
    def test_baseline_index_names(self) -> None:
        expected = {"PGM-index", "ALEX", "LIPP", "RMI", "RadixSpline",
                    "B-tree", "ART", "HOT", "FAST"}
        assert set(BASELINE_COMPARISON_DATA.keys()) == expected

    @pytest.mark.unit
    def test_baseline_pgm_values(self) -> None:
        pgm = BASELINE_COMPARISON_DATA["PGM-index"]
        assert pgm["eps_max"] == 47
        assert pgm["p50"] == 142
        assert pgm["p99"] == 312
        assert pgm["mem"] == 18.4
        assert pgm["build"] == 2.3

    @pytest.mark.unit
    def test_distribution_validation_count(self) -> None:
        assert len(DISTRIBUTION_VALIDATION_DATA) == 4

    @pytest.mark.unit
    def test_distribution_validation_cv_values(self) -> None:
        cv_map = {d["dataset"]: d["cv"] for d in DISTRIBUTION_VALIDATION_DATA}
        assert cv_map["amzn"] == 0.31
        assert cv_map["face"] == 0.72
        assert cv_map["osm"] == 1.85
        assert cv_map["wiki"] == 0.44

    @pytest.mark.unit
    def test_distribution_validation_ratios(self) -> None:
        for entry in DISTRIBUTION_VALIDATION_DATA:
            assert 0.80 <= entry["ratio"] <= 1.20

    @pytest.mark.unit
    def test_hpo_datasets(self) -> None:
        assert set(HPO_COMPARISON_DATA.keys()) == {"amzn", "osm"}

    @pytest.mark.unit
    def test_hpo_amzn_theory(self) -> None:
        theory = HPO_COMPARISON_DATA["amzn"]["theory"]
        assert theory["k"] == 1923
        assert theory["eps"] == 49.2
        assert theory["time_s"] == 12
        assert theory["evals"] == 1

    @pytest.mark.unit
    def test_practical_workflow_count(self) -> None:
        assert len(PRACTICAL_WORKFLOW_DATA) == 4

    @pytest.mark.unit
    def test_practical_workflow_deviations(self) -> None:
        for entry in PRACTICAL_WORKFLOW_DATA:
            assert entry["deviation_pct"] <= 5.0
