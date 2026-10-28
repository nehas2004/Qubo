"""
Simulated Bifurcation Machine (SBM) Solver.

Implements the Ballistic Simulated Bifurcation (bSB) algorithm (Goto et al. 2019/2021).
A high-speed quantum-inspired analog oscillator network solver running in NumPy.
"""

from typing import Optional, Dict, List, Any
import time
import numpy as np
import dimod

from src.dsl.ast_nodes import OptimizationProblem
from src.dsl.compiler import CompiledQUBO
from .base import BaseQUBOSolver, SolverResult


class SimulatedBifurcationSolver(BaseQUBOSolver):
    """
    Quantum-inspired Ballistic Simulated Bifurcation (bSB) solver.
    Vectorized simulation of adiabatic Kerr parametric oscillator networks.
    """

    def __init__(
        self,
        num_steps: int = 1000,
        num_shots: int = 30,          # Parallel particle ensemble
        dt: float = 0.5,
        seed: Optional[int] = 42,
    ):
        super().__init__(name="SimulatedBifurcation")
        self.num_steps = num_steps
        self.num_shots = num_shots
        self.dt = dt
        self.seed = seed

    def solve(
        self,
        problem: OptimizationProblem,
        compiled_qubo: Optional[CompiledQUBO] = None,
        penalty: Optional[float] = None,
        num_steps: Optional[int] = None,
        num_shots: Optional[int] = None,
        dt: Optional[float] = None,
        prefer_feasible: bool = True,
        **kwargs,
    ) -> SolverResult:
        """
        Solves problem using Ballistic Simulated Bifurcation on the compiled QUBO.
        """
        qubo = self._ensure_compiled(problem, compiled_qubo, penalty)
        n_vars = qubo.matrix.shape[0]

        start_time = time.perf_counter()

        if n_vars == 0:
            return self._decode_and_evaluate(problem, qubo, [], time.perf_counter() - start_time)

        steps = num_steps if num_steps is not None else self.num_steps
        shots = num_shots if num_shots is not None else self.num_shots
        step_size = dt if dt is not None else self.dt

        rng = np.random.default_rng(self.seed)

        # 1. Convert to Ising formulation: J matrix and h vector
        qubo_dict = qubo.to_dict()
        if not qubo_dict:
            qubo_dict = {(0, 0): 0.0}

        h_dict, j_dict, _ = dimod.qubo_to_ising(qubo_dict)

        h_arr = np.zeros(n_vars, dtype=float)
        for var_idx, h_val in h_dict.items():
            if var_idx < n_vars:
                h_arr[var_idx] = h_val

        # Symmetrized J matrix (minimizing sum J_ij s_i s_j + sum h_i s_i)
        J_mat = np.zeros((n_vars, n_vars), dtype=float)
        for (i, j), j_val in j_dict.items():
            if i < n_vars and j < n_vars:
                J_mat[i, j] = j_val
                J_mat[j, i] = j_val

        # Scaling factor xi for coupling: xi = 1.0 / (sqrt(N) * max(abs(J)))
        max_j = np.max(np.abs(J_mat)) if np.max(np.abs(J_mat)) > 0 else 1.0
        xi = 1.0 / (np.sqrt(n_vars) * max_j)

        # 2. Initialize ensemble of shots: positions x and momenta y
        # Shape: (shots, n_vars)
        x = rng.uniform(-0.1, 0.1, size=(shots, n_vars))
        y = np.zeros((shots, n_vars), dtype=float)

        a0 = 1.0

        # 3. Time evolution of the coupled Kerr parametric oscillator network
        for step in range(steps):
            # Control pump parameter p(t) ramped adiabatically from 0 to 1
            pt = (step / steps)

            # Equation of motion for momentum:
            # dy/dt = -(a0 - p(t)) * x - xi * (J @ x + h)
            # (Note minus sign because we seek minimum energy)
            coupling_term = (x @ J_mat) + h_arr  # shape (shots, n_vars)
            dy = - (a0 - pt) * x - xi * coupling_term

            y += step_size * dy
            x += step_size * (a0 * y)

            # Inelastic wall boundary condition at |x| = 1
            mask_upper = x > 1.0
            mask_lower = x < -1.0

            x[mask_upper] = 1.0
            y[mask_upper] = 0.0

            x[mask_lower] = -1.0
            y[mask_lower] = 0.0

        elapsed = time.perf_counter() - start_time

        # 4. Extract spins s_i = sign(x_i) and convert to binary bitstrings x_i = (s_i + 1) // 2
        spins = np.sign(x).astype(int)
        spins[spins == 0] = 1  # Break any rare exact zero ties
        binary_candidates = ((spins + 1) // 2).tolist()

        best_candidate: Optional[List[int]] = None
        best_candidate_energy: float = float("inf")
        best_feasible_candidate: Optional[List[int]] = None
        best_feasible_energy: float = float("inf")

        for bitstring in binary_candidates:
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
            "num_steps": steps,
            "num_shots": shots,
            "dt": step_size,
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
