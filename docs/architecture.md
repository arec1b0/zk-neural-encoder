# System Architecture

## Core Engineering Principles
*   **KISS & SOLID:** Minimal abstractions. The pipeline is designed for direct, linear execution without unnecessary wrapping.
*   **AOT Over JIT:** Encoding optimization happens Ahead-of-Time during the CI/CD compilation step. Zero runtime overhead on-chain.
*   **Strict Reproducibility:** Hermetic builds using `uv` lockfiles and complete Docker containerization.

## MLOps Lifecycle

The system operates as a self-contained pipeline:

1.  **Static Analysis:** The `StaticAnalyzer` extracts deterministic features (type, mutability, size) from target smart contract ABIs.
2.  **RL Optimization:** A Reinforcement Learning policy network evaluates layout strategies, optimizing for minimal algebraic constraints.
3.  **Experiment Tracking:** `MLflow` runs locally (`sqlite:///mlflow.db`) to version model weights, track reward convergence, and monitor degradation.
4.  **Manifest Export:** The optimized data layout is exported as a strict JSON manifest for the compiler toolchain.

## Infrastructure Stack
*   **Compute:** Python 3.11, PyTorch
*   **Package Management:** `uv`
*   **Deployment:** Docker Desktop (Windows 11 host), GitHub Actions