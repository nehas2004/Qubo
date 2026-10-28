"""
Classical Tabu Search Solver using dwave-tabu.
"""

from typing import Optional, Dict, List, Any
import time
import numpy as np
import tabu

from src.dsl.ast_nodes import OptimizationProblem
from src.dsl.compiler import CompiledQUBO
from .base import BaseQUBOSolver, SolverResult


class TabuSolver(BaseQUBOSolver):
    """
    Classical Tabu Search solver powered by D-Wave Tabu.
    Uses neighborhood local search with short-term tabu memory to prevent cycling.
    """

    def __init__(
        self,
        num_reads: int = 50,
        tenure: Optional[int] = None,
        timeout: Optional[int] = 50,  # Milliseconds per restart/read
        seed: Optional[int] = 42,
    ):
        super().__init__(name="TabuSearch")
        self.num_reads = num_reads
        self.tenure = tenure
        self.timeout = timeout
        self.seed = seed

    def solve(
        self,
        problem: OptimizationProblem,
        compiled_qubo: Optional[CompiledQUBO] = None,
        penalty: Optional[float] = None,
        num_reads: Optional[int] = None,
        tenure: Optional[int] = None,
        timeout: Optional[int] = None,
        prefer_feasible: bool = True,
        **kwargs,
    ) -> SolverResult:
        """
        Solves problem via Tabu Search on the compiled QUBO matrix.
        """
        qubo = self._ensure_compiled(problem, compiled_qubo, penalty)
        qubo_dict = qubo.to_dict()
        n_vars = qubo.matrix.shape[0]

        reads = num_reads if num_reads is not None else self.num_reads
        actual_tenure = tenure if tenure is not None else self.tenure
        actual_timeout = timeout if timeout is not None else self.timeout

        sampler = tabu.TabuSampler()

        start_time = time.perf_counter()

        if n_vars == 0:
            return self._decode_and_evaluate(problem, qubo, [], time.perf_counter() - start_time)

        if not qubo_dict:
            qubo_dict = {(0, 0): 0.0}

        sample_kwargs: Dict[str, Any] = {
            "num_reads": reads,
        }
        if actual_tenure is not None:
            sample_kwargs["tenure"] = actual_tenure
        if actual_timeout is not None:
            sample_kwargs["timeout"] = actual_timeout
        if self.seed is not None:
            sample_kwargs["seed"] = self.seed

        response = sampler.sample_qubo(qubo_dict, **sample_kwargs)
        elapsed = time.perf_counter() - start_time

        best_candidate: Optional[List[int]] = None
        best_candidate_energy: float = float("inf")
        best_feasible_candidate: Optional[List[int]] = None
        best_feasible_energy: float = float("inf")

        samples_list = []
        for sample_dict, energy in response.data(fields=["sample", "energy"]):
            bitstring = [int(sample_dict.get(i, 0)) for i in range(n_vars)]
            samples_list.append((bitstring, energy))

            if energy < best_candidate_energy:
                best_candidate_energy = energy
                best_candidate = bitstring

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
            "tenure": actual_tenure,
            "timeout_ms": actual_timeout,
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
