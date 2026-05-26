# SAGE Layer 1: EM (Enigma Machine Solver) — Complete Specification

**Status**: Locked  
**Version**: 1.2 (with all cutting-edge opportunities integrated at maximum intelligence)  
**Date**: May 26, 2026  

## 1. Vision & Philosophy

The EM is the **rich, verification-first, self-reflective solver engine** that generates high-signal, provenance-rich fragments while remaining lightweight for massive agent-native scaling.

**Core Principle**:  
**"Solve verifiably, reflect deeply, explore intelligently, and evolve continuously — turning every decision into maximum intelligence for the surrogate factory."**

## 2. Key Components

**2.1 EMInstance** (`sage-core/em/instance.py`)  
- Main entry point and orchestration loop with recursive and self-evolving capabilities.

**2.2 ContractEngine** (`sage-core/em/contract_engine.py`)  
- Generates verifiability contracts with hierarchical recursive decomposition and self-reflection.

**2.3 HybridReasoner** (`sage-core/em/hybrid_reasoner.py`)  
- Combines LLM debate with symbolic solvers (Marabou, LTN, SMT) and differentiable physics (JAX-FEM).

**2.4 IntuitionModule** (`sage-core/em/intuition_module.py`)  
- Temporal & physics-aware intuition using ESN/LSM/SNN reservoirs.

**2.5 ActiveExplorer** (`sage-core/em/active_explorer.py`)  
- Uncertainty-guided exploration and active learning during replan and scientist mode.

**2.6 Fragment** (`sage-core/em/fragment.py`)  
- Rich birth gate with dynamic EFS seed, uncertainty map, verifier trace, and self-reflection metadata.

## 3. Core Execution Loop (Rebuildable)

```python
def run_full_mission(self, challenge: str, verification_spec: str = "", goal_md: str = ""):
    guidance = synapse.get_guidance(challenge)
    
    contract = self.contract_engine.generate_verifiability_contract(
        challenge, goal_md, verification_spec, guidance
    )
    
    plan = self.plan_challenge(contract, guidance)
    
    while not self._smart_stop(plan):
        dvr_result = self.validation_oracle.run_dvr_pipeline(plan, contract)
        if not dvr_result.passed:
            plan = self.intelligent_replan(plan, dvr_result, mode="dvr_fail")
            continue
        
        outputs = self.orchestrate_subarbos(plan, contract)
        symbiosis = self.run_symbiosis(outputs)
        synthesis = self.hybrid_reasoner.synthesis_arbos(outputs, symbiosis, contract)
        
        validation = self.validation_oracle.validate(synthesis, contract)
        
        if self.active_explorer.needs_replan_or_exploration(validation):
            plan = self.intelligent_replan(plan, validation, mode="scientist")
            continue
        
        self._end_of_loop_processing(validation)
    
    self._end_of_run_debrief()
