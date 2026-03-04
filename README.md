# PAC-Index: How Much Training Data Do Learned Indexes Need?

A formal PAC learning theory framework for analyzing sample complexity and generalization guarantees of learned database indexes. This repository provides the complete implementation for computing VC dimension bounds, sample complexity theorems, and theory-guided hyperparameter selection across all major learned index architectures.

## Overview

Learned indexes replace traditional tree-based data structures with machine learning models that predict the position of keys in sorted data. While they achieve substantial practical performance gains, they have lacked formal theoretical foundations regarding **how much training data is needed** for reliable predictions.

PAC-Index addresses this gap by applying Probably Approximately Correct (PAC) learning theory to derive:

- **VC dimension bounds** for piecewise linear (PWL), RMI, ALEX, LIPP, and RadixSpline architectures
- **Sample complexity theorems**: m = O((d_VC + log(1/delta)) / epsilon^2)
- **Distribution-dependent refinements** that tighten bounds by 1/CV^2
- **Theory-guided hyperparameter selection** requiring zero index builds

## Key Results

### VC Dimension Bounds

| Architecture | VC Dimension | Bound Type |
|:-------------|:-------------|:-----------|
| PWL (k segments) | Theta(k) | Tight: 2k <= VC <= 3k-1 |
| RMI (d levels, w branching) | O(d * w^d * log(wd)) | Upper bound |
| ALEX (n keys, epsilon error) | O(n / epsilon) | Upper bound |
| LIPP (n keys) | O(n) | Upper bound |
| RadixSpline (r bits, s splines) | O(2^r * s) | Upper bound |

### Theory-Practice Correlation

Experimental validation on SOSD benchmarks shows Pearson correlation r = 0.94 (p < 0.001) between theoretical predictions and observed performance, with empirical convergence rate alpha in [0.48, 0.52] matching the predicted O(1/sqrt(m)).

### Query Latency (PGM-index, P50 in ns)

| Training Size | amzn | face | osm | wiki |
|--------------:|-----:|-----:|----:|-----:|
| 10^4 | 892 | 967 | 1234 | 1012 |
| 10^5 | 356 | 412 | 623 | 445 |
| 10^6 | 198 | 234 | 345 | 256 |
| 10^7 | 142 | 167 | 223 | 178 |
| 10^8 | 121 | 134 | 178 | 145 |

### Baseline Comparison (amzn, m = 10^7)

| Index | eps_max | P50 (ns) | P99 (ns) | Mem (MB) | Build (s) |
|:------|--------:|---------:|---------:|---------:|----------:|
| PGM-index | 47 | 142 | 312 | 18.4 | 2.3 |
| ALEX | 52 | 156 | 334 | 24.1 | 4.7 |
| LIPP | 0 | 89 | 156 | 892.4 | 8.2 |
| RMI (2-level) | 78 | 178 | 389 | 12.3 | 1.8 |
| RadixSpline | 63 | 167 | 356 | 21.7 | 0.9 |
| B-tree (STX) | --- | 312 | 589 | 3200.0 | 45.2 |
| ART | --- | 198 | 423 | 1456.0 | 28.4 |
| HOT | --- | 167 | 389 | 892.0 | 34.1 |
| FAST | --- | 223 | 478 | 2100.0 | 12.3 |

### Distribution Validation

| Dataset | CV | Gap rho | Predicted | Observed | Ratio |
|:--------|---:|--------:|----------:|---------:|------:|
| amzn | 0.31 | 0.08 | 0.096/sqrt(m) | 0.089/sqrt(m) | 1.08 |
| face | 0.72 | 0.12 | 0.518/sqrt(m) | 0.551/sqrt(m) | 0.94 |
| osm | 1.85 | 0.31 | 3.423/sqrt(m) | 3.891/sqrt(m) | 0.88 |
| wiki | 0.44 | 0.15 | 0.194/sqrt(m) | 0.212/sqrt(m) | 0.92 |

### HPO Comparison

| Method | amzn k* | amzn eps | Time | Evals | osm k* | osm eps | Time | Evals |
|:-------|--------:|---------:|-----:|------:|-------:|--------:|-----:|------:|
| Theory-guided | 1,923 | 49.2 | 12s | 1 | 68,450 | 54.3 | 18s | 1 |
| Bayesian opt | 1,847 | 47.8 | 28m | 50 | 71,023 | 51.9 | 42m | 50 |
| Random search | 2,341 | 52.1 | 15m | 100 | 58,234 | 61.4 | 23m | 100 |
| Grid search | 1,856 | 48.1 | 2.1h | 9 | 71,234 | 52.1 | 4.8h | 9 |

Theory-guided selection achieves within 2-5% of Bayesian optimization accuracy with **zero index builds**.

### Practical Workflow

| Dataset | Target eps | k*(theory) | k*(empirical) | Dev (%) | Theory | Grid |
|:--------|----------:|-----------:|--------------:|--------:|-------:|-----:|
| amzn | 100 | 1,923 | 1,856 | 3.6 | 12s | 2.1h |
| face | 100 | 10,368 | 10,789 | 3.9 | 14s | 3.4h |
| osm | 100 | 68,450 | 71,234 | 3.9 | 18s | 4.8h |
| wiki | 50 | 15,488 | 15,200 | 1.9 | 13s | 2.8h |

## Installation

### Requirements

- Python >= 3.10, < 3.12
- 8+ GB RAM (32+ GB for full 200M datasets)
- Ubuntu 22.04+ recommended

### Quick Start

```bash
# Clone the repository
git clone https://github.com/bettyguo/pac-learned-index.git
cd pac-learned-index

# Install with development dependencies
pip install -e ".[all]"

# Download SOSD benchmark data
make benchmarks

# Run quick demo
python demo.py

# Run theory-guided analysis
python run_system.py

# Run all experiments
make run-all
```

### Installation Options

```bash
# Minimal installation
pip install -e .

# With development tools (testing, linting, type checking)
pip install -e ".[dev]"

# Full installation with all extras
pip install -e ".[all]"
```

## Repository Structure

```
pac_index/
├── configs/                 # YAML configuration files
│   ├── default.yaml         # Default configuration
│   ├── debug.yaml           # Quick debug configuration
│   ├── datasets/            # Per-dataset configs with distribution properties
│   ├── experiment/          # Experiment-specific configs
│   └── workloads/           # Benchmark workload definitions
├── src/pac_index/           # Core package source
│   ├── core/                # Engine, config, pipeline
│   ├── evaluation/          # Metrics, statistics, visualization
│   ├── storage/             # Index structures and data management
│   ├── query/               # Query processing
│   ├── workload/            # Workload generation and benchmarking
│   └── utils/               # Reproducibility, I/O, logging
├── tests/                   # Test suite
├── run_system.py            # Main analysis entry point
├── run_experiment.py        # Experiment runner
└── demo.py                  # Interactive demo
```

## Experiments

### Hardware Configuration

All experiments were conducted on:
- **CPU**: Intel Xeon Gold 6248R (3.0 GHz)
- **RAM**: 384 GB DDR4
- **OS**: Ubuntu 22.04
- **Compiler**: GCC 11.4 with C++17, -O3 optimization
- **Seeds**: 42, 123, 256, 314, 512, 628, 729, 841, 953, 1024

### Datasets (SOSD Benchmark)

| Dataset | Size | Key Type | CDF Type | CV |
|:--------|-----:|:---------|:---------|---:|
| amzn | 200M | uint64 | smooth | 0.31 |
| face | 200M | uint64 | near-uniform | 0.72 |
| osm | 200M | uint64 | erratic | 1.85 |
| wiki | 200M | uint64 | stepped | 0.44 |

### Running Individual Experiments

```bash
# Main comparison (all indexes, all datasets)
make run-main

# Scalability study
make run-scalability

# Ablation study
make run-ablation

# Collect and analyze results
make eval

```

## Testing

```bash
# Run all tests
make test

# Fast tests only (no data dependencies)
make test-fast

# With coverage report
make test-cov

# Full quality checks (lint + typecheck + test)
make quality
```

## Development

```bash
# Install with dev dependencies
make install-dev

# Format code
make format

# Run linter
make lint

# Type checking
make typecheck
```
