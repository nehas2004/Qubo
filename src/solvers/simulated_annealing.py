"""
Classical Simulated Annealing (SA) Solver using dwave-neal.
"""

from typing import Optional, Dict, List, Any
import time
import numpy as np
import neal

from src.dsl.ast_nodes import OptimizationProblem
from src.dsl.compiler import CompiledQUBO
from .base import BaseQUBOSolver, SolverResult


class SimulatedAnnealingSolver(BaseQUBOSolver):
    """
    Classical Simulated Annealing solver powered by D-Wave Neal.
    Uses thermal Markov Chain Monte Carlo (MCMC) with geometric temperature cooling.
    """

    def __init__(
        self,
        num_reads: int = 100,
        num_sweeps: int = 1000,
        beta_range: Optional[List[float]] = None,
        seed: Optional[int] = 42,
    ):
        super().__init__(name="SimulatedAnnealing")
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.beta_range = beta_range
        self.seed = seed

    def solve(
        self,
        problem: OptimizationProblem,
        compiled_qubo: Optional[CompiledQUBO] = None,
        penalty: Optional[float] = None,
        num_reads: Optional[int] = None,
        num_sweeps: Optional[int] = None,
        prefer_feasible: bool = True,
        **kwargs,
    ) -> SolverResult:
        """
        Solves problem via Simulated Annealing on the compiled QUBO matrix.
        """
        qubo = self._ensure_compiled(problem, compiled_qubo, penalty)
        qubo_dict = qubo.to_dict()
        n_vars = qubo.matrix.shape[0]

        reads = num_reads if num_reads is not None else self.num_reads
        sweeps = num_sweeps if num_sweeps is not None else self.num_sweeps

        sampler = neal.SimulatedAnnealingSampler()

        start_time = time.perf_counter()

        # Handle trivial zero-variable or empty matrix
        if n_vars == 0:
            return self._decode_and_evaluate(problem, qubo, [], time.perf_counter() - start_time)

        # If qubo_dict is empty (no non-zero entries), create a zero-entry to allow dimod sampling
        if not qubo_dict:
            qubo_dict = {(0, 0): 0.0}

        sample_kwargs: Dict[str, Any] = {
            "num_reads": reads,
            "num_sweeps": sweeps,
        }
        if self.seed is not None:
            sample_kwargs["seed"] = self.seed
        if self.beta_range is not None:
            sample_kwargs["beta_range"] = self.beta_range

        response = sampler.sample_qubo(qubo_dict, **sample_kwargs)
        elapsed = time.perf_counter() - start_time

        # Extract samples sorted by energy
        best_candidate: Optional[List[int]] = None
        best_candidate_energy: float = float("inf")
        best_feasible_candidate: Optional[List[int]] = None
        best_feasible_energy: float = float("inf")

        samples_list = []
        for sample_dict, energy in response.data(fields=["sample", "energy"]):
            # Build full bitstring vector of length n_vars
            bitstring = [int(sample_dict.get(i, 0)) for i in range(n_vars)]
            samples_list.append((bitstring, energy))

            if energy < best_candidate_energy:
                best_candidate_energy = energy
                best_candidate = bitstring

            # Check feasibility if requested
            if prefer_feasible:
                eval_res = self._decode_and_evaluate(problem, qubo, bitstring, 0.0)
                if eval_res.is_feasible and energy < best_feasible_energy:
                    best_feasible_energy = energy
                    best_feasible_candidate = bitstring

        chosen_bitstring = best_feasible_candidate if (prefer_feasible and best_feasible_candidate is not None) else best_candidate
        if chosen_bitstring is None:
            chosen_bitstring = [0] * n_vars

        solver_info = {
            "num_reads": reads,
            "num_sweeps": sweeps,
            "found_feasible": best_feasible_candidate is not None,
            "best_energy_seen": best_candidate_energy + qubo.constant_offset,
            "unique_samples": len(samples_list),
        }

        return self._decode_and_evaluate(
            problem,
            qubo,
            chosen_bitstring,
            elapsed,
            status="FEASIBLE" if (best_feasible_candidate is not None) else "UNKNOWN",
            solver_info=solver_info,
        )
