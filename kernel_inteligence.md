✅ Layer: Compute Self-Optimization & Kernel Intelligence — Complete Detailed Specification
This document provides a comprehensive, rebuildable specification for kernel optimization across the entire SAGE system. It documents where and how kernel optimization is used, why it is integrated that way, and how it synergizes with surrogate/PINO work.
1. Vision & Philosophy
Compute Self-Optimization (Objective 7) means SAGE continuously learns and improves its own low-level CUDA kernels, memory strategies, and hardware utilization in tight synergy with surrogate design and verification.
The goal is a self-improving compute-surrogate loop: better kernels → faster & more accurate verification/surrogates → richer fragments → better kernel strategies.
2. Core Design Decisions
•  Cross-cutting concern led by Layer 3 (Synapse Meta-RL), executed in Layers 1, 2, and 4.
•  Agentic + Surrogate-Guided: Use Meta-RL and small performance surrogates to search kernel space efficiently.
•  Hardware-Aware: Profiles and decisions adapt to available GPUs, memory, and tensor shapes.
•  Safety & Fallback: Always maintain CPU paths and validation guards.
3. Integration by Layer
Layer 1: EM (Execution)
•  Usage: Offload PINO residual checks, hybrid tensor contractions, and validation steps to optimized kernels when GPU is available (profile-driven).
•  Specifics:
	•  CUDA Graphs for repetitive validation passes.
	•  Fused FFT + spectral multiply + residual kernels (cuTensor + custom kernels).
	•  Dynamic kernel selection from Synapse policy.
•  Documentation Point: EM calls kernel_manager.execute_pino_verification(...) which routes to the best learned kernel.
Layer 2: iOS (Intelligence Factory)
•  Usage: GPU-aware swarm orchestration and profile calibration.
•  Specifics:
	•  Detect available GPUs and adjust batch sizes, parallel EM instances, and profile selection based on kernel performance history.
	•  Calibration flight tests include kernel performance metrics.
	•  Challenge pulling prioritizes GPU-friendly (PINO-heavy) challenges when optimized kernels are available.
•  Documentation Point: orchestrator.py queries Synapse for current kernel policy before launching swarms.
Layer 3: Synapse (Meta-RL & Distillation) — Primary Home
•  Usage: Meta-RL policy network learns kernel optimization strategies.
•  Specifics:
	•  Kernel Policy Agent (agentic RL or surrogate-guided search) proposes, tests, and evolves CUDA kernel configurations.
	•  Surrogate-Guided Search: Small performance surrogates predict kernel runtime/memory for candidate configurations, reducing real GPU evaluations.
	•  Distillation of “kernel recipes” into specialized surrogates (e.g., “best kernel set for quantum PINO”).
	•  Feedback: Real kernel performance + surrogate accuracy + fragment yield → Meta-RL reward.
•  Key Algorithms:
	•  cotengra for contraction path optimization.
	•  CUDA Agent-style iterative evolution for custom kernels.
	•  Differentiable kernel approximation for gradient-based tuning.
Layer 4: Validation Oracle
•  Usage: All physics-heavy verification uses optimized kernels.
•  Specifics:
	•  Hybrid PINO + Tensor Network verifier uses cuTensorNet + cotengra-optimized paths executed via CUDA Graphs.
	•  Multi-fidelity discrepancy and Bayesian PINO use fused custom kernels.
	•  Caching of kernel results for repeated subtask patterns.
•  Documentation Point: Every validate() call logs kernel version and performance for Meta-RL.
4. Synergistic Loop (The Moat)
1.  Synapse proposes new kernel configuration.
2.  iOS launches targeted EM swarms using the kernel.
3.  EM + Oracle execute verification with the kernel and record performance.
4.  Fragments + telemetry flow back to Synapse.
5.  Meta-RL updates kernel policy and surrogate design.
6.  Repeat — creating compounding gains in speed, accuracy, and capability.
5. Implementation Roadmap & Documentation Points
•  KernelManager (new cross-layer module in Synapse): Central registry and dispatcher for learned kernels.
•  Performance Surrogate: Small model that predicts kernel metrics to accelerate search.
•  Telemetry: All kernel runs log FLOPs, memory bandwidth, occupancy, and end-to-end time.
•  Fallback: CPU paths always available with clear performance degradation signals.
This specification ensures kernel optimization is systematic, documented, and synergistic with surrogate/PINO work.
