"""
Simulated Quantum Annealing (SQA) Solver.

Uses Path-Integral Monte Carlo (PIMC) with Suzuki-Trotter decomposition
to simulate transverse-field quantum annealing (quantum tunneling).
"""

from typing import Optional, Dict, List, Any, Tuple
import time
import math
import numpy as np
import dimod

from src.dsl.ast_nodes import OptimizationProblem
from src.dsl.compiler import CompiledQUBO
from .base import BaseQUBOSolver, SolverResult


class SimulatedQuantumAnnealingSolver(BaseQUBOSolver):
    """
    Path-Integral Monte Carlo (PIMC) Simulated Quantum Annealing solver.
    Simulates transverse-field quantum tunneling across M Trotter replicas.
    """

    def __init__(
        self,
        num_trotter_slices: int = 8,
        num_steps: int = 200,
        beta: float = 5.0,              # Inverse physical temperature 1 / (k_B * T)
        gamma_initial: float = 3.0,     # Initial transverse field
        gamma_final: float = 0.001,     # Final transverse field
        seed: Optional[int] = 42,
    ):
        super().__init__(name="SimulatedQuantumAnnealing")
        self.num_trotter_slices = num_trotter_slices
        self.num_steps = num_steps
        self.beta = beta
        self.gamma_initial = gamma_initial
        self.gamma_final = gamma_final
        self.seed = seed

    def solve(
        self,
        problem: OptimizationProblem,
        compiled_qubo: Optional[CompiledQUBO] = None,
        penalty: Optional[float] = None,
        num_trotter_slices: Optional[int] = None,
        num_steps: Optional[int] = None,
        beta: Optional[float] = None,
        prefer_feasible: bool = True,
        **kwargs,
    ) -> SolverResult:
        """
        Solves problem using Path-Integral Monte Carlo Simulated Quantum Annealing.
        """
        qubo = self._ensure_compiled(problem, compiled_qubo, penalty)
        n_vars = qubo.matrix.shape[0]

        start_time = time.perf_counter()

        if n_vars == 0:
            return self._decode_and_evaluate(problem, qubo, [], time.perf_counter() - start_time)

        M = num_trotter_slices if num_trotter_slices is not None else self.num_trotter_slices
        steps = num_steps if num_steps is not None else self.num_steps
        inv_temp = beta if beta is not None else self.beta

        rng = np.random.default_rng(self.seed)

        # 1. Convert QUBO to Ising formulation: s_i in {-1, +1}
        # H_ising = sum h_i s_i + sum_{i < j} J_ij s_i s_j + offset
        qubo_dict = qubo.to_dict()
        if not qubo_dict:
            qubo_dict = {(0, 0): 0.0}

        h_dict, j_dict, ising_offset = dimod.qubo_to_ising(qubo_dict)

        h_arr = np.zeros(n_vars, dtype=float)
        for var_idx, h_val in h_dict.items():
            if var_idx < n_vars:
                h_arr[var_idx] = h_val

        # Symmetric interaction matrix J
        J_mat = np.zeros((n_vars, n_vars), dtype=float)
        for (i, j), j_val in j_dict.items():
            if i < n_vars and j < n_vars:
                J_mat[i, j] = j_val
                J_mat[j, i] = j_val

        # 2. Initialize M Trotter replicas with random spins {-1, +1}
        # Shape: (M, n_vars)
        spins = rng.choice([-1, 1], size=(M, n_vars)).astype(np.int8)

        # Linear schedule for transverse field Gamma(t)
        gamma_values = np.linspace(self.gamma_initial, self.gamma_final, steps)

        for step_idx in range(steps):
            gamma = float(gamma_values[step_idx])

            # Inter-slice Trotter coupling:
            # J_perp = -1 / (2 * beta) * ln(tanh(beta * gamma / M))
            arg = max(1e-12, min(1.0 - 1e-12, math.tanh((inv_temp * gamma) / M)))
            J_perp = -0.5 * (1.0 / inv_temp) * math.log(arg)

            # Perform a sweep over all slices and variables
            for m in range(M):
                m_prev = (m - 1) % M
                m_next = (m + 1) % M

                # Local effective field on slice m from classical Ising interactions
                # local_field_i = h_i + sum_j J_ij * s_j
                local_field = h_arr + (J_mat @ spins[m])

                for i in range(n_vars):
                    s_current = spins[m, i]

                    # Intra-slice energy change for flipping spin i:
                    # Delta E_intra = -2 * s_i * (h_i + sum_j J_ij s_j) / M
                    dE_intra = -(2.0 * s_current * local_field[i]) / M

                    # Inter-slice energy change (coupling to nearest Trotter neighbors):
                    # -J_perp * s_m * (s_{m-1} + s_{m+1})
                    # Delta E_inter = 2 * J_perp * s_m * (s_{m-1} + s_{m+1})
                    neighbor_sum = spins[m_prev, i] + spins[m_next, i]
                    dE_inter = 2.0 * J_perp * s_current * neighbor_sum

                    dE = dE_intra + dE_inter

                    # Metropolis-Hastings acceptance condition: exp(-beta * dE)
                    if dE < 0 or rng.random() < math.exp(-inv_temp * dE):
                        spins[m, i] = -s_current
                        # Update local field vector for current slice
                        delta_spin = spins[m, i] - s_current
                        local_field += J_mat[:, i] * delta_spin

        elapsed = time.perf_counter() - start_time

        # 3. Evaluate the classical energy of all M Trotter slices
        best_candidate: Optional[List[int]] = None
        best_candidate_energy: float = float("inf")
        best_feasible_candidate: Optional[List[int]] = None
        best_feasible_energy: float = float("inf")

        for m in range(M):
            # Convert spins s in {-1, 1} -> binary x in {0, 1}: x_i = (s_i + 1) // 2
            bitstring = [1 if spins[m, i] == 1 else 0 for i in range(n_vars)]
            e = qubo.energy(bitstring)

            if e < best_candidate_energy:
                best_candidate_energy = e
                best_candidate = bitstring

            if prefer_feasible:
                eval_res = self._decode_and_evaluate(problem, qubo, bitstring, 0.0)
                if eval_res.is_feasible and e < best_feasible_energy:
                    best_feasible_energy = e
                    best_feasible_candidate = bitstring

        chosen_bitstring = best_feasible_candidate if (prefer_feasible and best_feasible_candidate is not None) else best_candidate
        if chosen_bitstring is None:
            chosen_bitstring = [0] * n_vars

        solver_info = {
            "num_trotter_slices": M,
            "num_steps": steps,
            "beta": inv_temp,
            "found_feasible": best_feasible_candidate is not None,
            "best_energy_seen": best_candidate_energy,
        }

        return self._decode_and_evaluate(
            problem,
            qubo,
            chosen_bitstring,
            elapsed,
            status="FEASIBLE" if (best_feasible_candidate is not None) else "UNKNOWN",
            solver_info=solver_info,
        )
