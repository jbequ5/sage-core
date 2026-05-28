flowchart TD
    subgraph EM["Layer 1 — EM (Enigma Machine)"]
        A[ContractEngine + HybridReasoner] --> B[FragmentFactory Birth Gate]
        B --> C[Benchmark Scoring Gate v1.2.1]
        C --> D[DiscrepancyVector + Geodesic Premonition]
        D --> E[Re-loop Decision]
    end

    subgraph iOS["Layer 2 — iOS"]
        F[Calibration + KAS Hunts] --> G[ANIL Warm-Start → Multi-Fidelity BO]
        G --> H[Team Composition: MoPE × MoDE × PINO]
        H --> I[ElasticScheduler + Yield Tracking]
    end

    subgraph Synapse["Layer 3 — Synapse (Objective Flow Engine)"]
        J[NeurELA Riemannian+Hyperbolic] --> K[Temporal Causal PLON]
        K --> L[Hyperbolic Causal Geodesics]
        L --> M[MetaRLLoop: RND + ANIL]
        M --> N[PINO Bank FNO-Transformer + UQ]
        N --> O[Non-Stationarity: EWC + Drift + RND Early Warning]
        O --> P[Objective Flow Engine Routing]
    end

    subgraph L4["Layer 4 — Validation Oracle"]
        Q[Bayesian Aggregation weighted by PINO UQ] --> R[Self-Evolving Oracles]
    end

    %% The 8 Intelligent Integrations (green edges)
    M -.->|1. RND Uncertainty → PINO UQ Prioritization| N
    M -.->|2. ANIL Warm-Start| G
    O -.->|3. Drift → Targeted PINO Refresh| N
    J -.->|4. Geodesic Premonition Biases BO| G
    O -.->|5. EWC weighted by ContributionScore| K
    N -.->|6. PINO UQ → Oracle Weighting| Q
    M -.->|7. RND Error → Early Drift Signal| O
    L -.->|8. Hyperbolic Causal Geodesics| K

    %% Main Data Flow
    E -->|Rich Fragments + Discrepancy| Synapse
    I -->|Team Performance Telemetry| M
    P -->|Guidance + Distilled Specialists + Kernels| EM
    P -->|Updated Teams + PINO| iOS
    R -->|Verification Feedback| Synapse

    classDef core fill:#1e3a8a,stroke:#3b82f6,color:white
    classDef integration fill:#166534,stroke:#4ade80,color:white
    classDef ofe fill:#581c87,stroke:#c026ff,color:white

    class EM,iOS,Synapse,L4 core
    class M,N,O,P integration
    class L,G integration
    class P ofe
