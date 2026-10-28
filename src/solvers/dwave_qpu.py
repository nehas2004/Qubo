"""
D-Wave Quantum Hardware (QPU) Solver interface.

Supports connecting to real D-Wave Advantage QPUs via dwave-system / Leap API,
with graceful fallback/diagnostics when hardware tokens are unconfigured.
"""

import os
from typing import Optional, Dict, List, Any
import time

from src.dsl.ast_nodes import OptimizationProblem
from src.dsl.compiler import CompiledQUBO
from .base import BaseQUBOSolver, SolverResult


class DWaveQPUSolver(BaseQUBOSolver):
    """
    Quantum Processing Unit (QPU) solver using D-Wave Advantage System.
    Requires dwave-system and a valid D-Wave Leap API token.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        solver_name: Optional[str] = None,
        num_reads: int = 100,
        annealing_time: Optional[float] = None,  # Microseconds (default: 20us)
    ):
        super().__init__(name="DWaveQPU")
        self.token = token or os.environ.get("DWAVE_API_TOKEN")
        self.target_solver = solver_name
        self.num_reads = num_reads
        self.annealing_time = annealing_time

    def solve(
        self,
        problem: OptimizationProblem,
        compiled_qubo: Optional[CompiledQUBO] = None,
        penalty: Optional[float] = None,
        num_reads: Optional[int] = None,
        token: Optional[str] = None,
        prefer_feasible: bool = True,
        **kwargs,
    ) -> SolverResult:
        """
        Submits the compiled QUBO to the D-Wave Quantum Processing Unit (QPU).
        """
        qubo = self._ensure_compiled(problem, compiled_qubo, penalty)
        qubo_dict = qubo.to_dict()
        n_vars = qubo.matrix.shape[0]

        start_time = time.perf_counter()

        if n_vars == 0:
            return self._decode_and_evaluate(problem, qubo, [], time.perf_counter() - start_time)

        api_token = token or self.token
        reads = num_reads if num_reads is not None else self.num_reads

        # Check for dwave-system library
        try:
            from dwave.system import DWaveSampler, EmbeddingComposite
        except ImportError:
            return SolverResult(
                solver_name=self.name,
                status="ERROR",
                is_feasible=False,
                objective_value=None,
                qubo_energy=None,
                wall_time=time.perf_counter() - start_time,
                assignment={},
                num_violations=len(problem.constraints),
                solver_info={
                    "error": (
                        "Package 'dwave-system' is not installed in the environment. "
                        "Install with: pip install dwave-system"
                    ),
                    "setup_guide": "Sign up at cloud.dwavesys.com/leap to obtain an API token."
                },
            )

        # Check for Leap API token
        if not api_token:
            return SolverResult(
                solver_name=self.name,
                status="ERROR",
                is_feasible=False,
                objective_value=None,
                qubo_energy=None,
                wall_time=time.perf_counter() - start_time,
                assignment={},
                num_violations=len(problem.constraints),
                solver_info={
                    "error": (
                        "No D-Wave Leap API token detected. Set the DWAVE_API_TOKEN environment variable "
                        "or pass token='...' to DWaveQPUSolver."
                    ),
                    "setup_guide": "Sign up at cloud.dwavesys.com/leap to obtain an API token."
                },
            )

        try:
            sampler_kwargs = {"token": api_token}
            if self.target_solver:
                sampler_kwargs["solver"] = self.target_solver

            qpu_sampler = DWaveSampler(**sampler_kwargs)
            composite_sampler = EmbeddingComposite(qpu_sampler)

            sample_kwargs: Dict[str, Any] = {"num_reads": reads}
            if self.annealing_time is not None:
                sample_kwargs["annealing_time"] = self.annealing_time

            response = composite_sampler.sample_qubo(qubo_dict, **sample_kwargs)
            elapsed = time.perf_counter() - start_time

            best_candidate: Optional[List[int]] = None
            best_candidate_energy: float = float("inf")
            best_feasible_candidate: Optional[List[int]] = None
            best_feasible_energy: float = float("inf")

            for sample_dict, energy in response.data(fields=["sample", "energy"]):
                bitstring = [int(sample_dict.get(i, 0)) for i in range(n_vars)]
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

            qpu_timing = response.info.get("timing", {})

            return self._decode_and_evaluate(
                problem,
                qubo,
                chosen_bitstring,
                elapsed,
                status="FEASIBLE" if (best_feasible_candidate is not None) else "UNKNOWN",
                solver_info={
                    "qpu_timing": qpu_timing,
                    "num_reads": reads,
                    "found_feasible": best_feasible_candidate is not None,
                    "best_energy_seen": best_candidate_energy + qubo.constant_offset,
                },
            )

        except Exception as e:
            return SolverResult(
                solver_name=self.name,
                status="ERROR",
                is_feasible=False,
                objective_value=None,
                qubo_energy=None,
                wall_time=time.perf_counter() - start_time,
                assignment={},
                num_violations=len(problem.constraints),
                solver_info={"exception": str(e)},
            )
