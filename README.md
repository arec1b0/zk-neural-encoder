# ZK Neural Encoder

[![CI](https://github.com/arec1b0/zk-neural-encoder/actions/workflows/ci.yml/badge.svg)](https://github.com/arec1b0/zk-neural-encoder/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An Ahead-of-Time (AOT) neural compiler designed to optimize Zero-Knowledge (ZK) proof generation for Starknet developers.

Current ZK compilers often rely on static, naive heuristics for state variable data layouts, leading to bloated circuit constraints. **ZK Neural Encoder** replaces these hardcoded rules with a Reinforcement Learning (RL) agent that parses Cairo ABIs and dynamically determines the most cost-efficient encoding strategy (e.g., Limb Decomposition vs. Single Field). The result is a minimized algebraic constraint footprint, lowering execution time and prover overhead.

## Architecture & Stack

Built with a strict focus on MLOps best practices and high-concurrency production readiness:

* **Core:** Python 3.11, PyTorch (RL Agent)
* **Tracking & Lifecycle:** MLflow (Experiment tracking, model artifact versioning)
* **Package Management:** `uv` (Fast dependency resolution)
* **Deployment:** Docker, GitHub Actions CI/CD

## Quick Start

### Local Execution (via `uv`)

1. **Install `uv`** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Run the pipeline directly** (this automatically syncs dependencies and isolates the runtime environment):
   ```bash
   uv run python run_pipeline.py
   ```

### Docker Execution

Run the completely containerized pipeline without local Python dependencies:

```bash
docker build -t zk-neural-encoder .
docker run --rm zk-neural-encoder uv run python run_pipeline.py
```

## Development & Testing

This project uses `uv` for development dependency management and `pytest` for the testing suite.

```bash
# Sync all dependencies including dev-groups
uv sync --all-extras --dev

# Run the test suite
uv run pytest tests/
```

## Output Manifest

The pipeline outputs a deterministic JSON manifest (`encoding_manifest.json`) designed for zero-overhead integration directly into Cairo compilation toolchains:

```json
{
  "version": "1.0",
  "generator": "Neural-Guided-AOT",
  "layouts": {
    "totalSupply": {
      "type": "uint256",
      "encoding_strategy": "LIMB_DECOMPOSITION",
      "estimated_constraints": 12
    },
    "owner": {
      "type": "address",
      "encoding_strategy": "LIMB_DECOMPOSITION",
      "estimated_constraints": 3
    }
  },
  "summary": {
    "total_estimated_constraints": 27,
    "variable_count": 3
  }
}
```

## License

Distributed under the MIT License.