"""
Base abstractions, result containers, and helper methods for QUBO and optimization solvers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import time
import numpy as np

from src.dsl.ast_nodes import OptimizationProblem, VarRef
from src.dsl.compiler import CompiledQUBO, QUBOCompiler
from src.dsl.checker import check_solution_feasibility, FeasibilityResult


@dataclass
class SolverResult:
    """Standardized result container returned by all solvers."""
    solver_name: str
    status: str                         # "OPTIMAL", "FEASIBLE", "INFEASIBLE", "TIMEOUT", "ERROR"
    is_feasible: bool
    objective_value: Optional[float]    # Original problem objective value
    qubo_energy: Optional[float]        # Lowered QUBO energy x^T Q x + constant
    wall_time: float                    # Execution time in seconds
    assignment: Dict[str, float]        # Original variable assignments (e.g. {"x[0]": 1.0, ...})
    raw_bitstring: Optional[List[int]] = None  # Full binary vector including slacks
    num_violations: int = 0
    max_violation: float = 0.0
    solver_info: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        obj_str = f"{self.objective_value:.4f}" if self.objective_value is not None else "N/A"
        energy_str = f"{self.qubo_energy:.4f}" if self.qubo_energy is not None else "N/A"
        feas_str = "FEASIBLE" if self.is_feasible else f"INFEASIBLE ({self.num_violations} violations)"
        return (
            f"[{self.solver_name}] status={self.status} | {feas_str} | "
            f"objective={obj_str} | qubo_energy={energy_str} | time={self.wall_time*1000:.2f}ms"
        )


class BaseQUBOSolver(ABC):
    """Abstract base class for all optimization solvers."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def solve(
        self,
        problem: OptimizationProblem,
        compiled_qubo: Optional[CompiledQUBO] = None,
        **kwargs,
    ) -> SolverResult:
        """Solve an OptimizationProblem or CompiledQUBO instance."""
        pass

    def _ensure_compiled(
        self,
        problem: OptimizationProblem,
        compiled_qubo: Optional[CompiledQUBO] = None,
        penalty: Optional[float] = None,
    ) -> CompiledQUBO:
        """Ensures a CompiledQUBO object is available."""
        if compiled_qubo is not None:
            return compiled_qubo
        compiler = QUBOCompiler(default_penalty=penalty)
        return compiler.compile(problem)

    def _decode_and_evaluate(
        self,
        problem: OptimizationProblem,
        compiled_qubo: CompiledQUBO,
        bitstring: List[int],
        wall_time: float,
        status: str = "FEASIBLE",
        solver_info: Optional[Dict[str, Any]] = None,
    ) -> SolverResult:
        """
        Decodes a full binary bitstring (including slack variables) into original variable
        assignments, evaluates problem constraints for feasibility, and computes objective.
        """
        assignment: Dict[str, float] = {}
        orig_var_names = {ref.to_string() for ref in problem.get_all_var_refs()}

        for idx, val in enumerate(bitstring):
            if idx in compiled_qubo.index_to_var:
                var_name = compiled_qubo.index_to_var[idx]
                if var_name in orig_var_names:
                    assignment[var_name] = float(val)

        # Evaluate problem-level feasibility & objective
        feas_result: FeasibilityResult = check_solution_feasibility(problem, assignment)

        # Compute QUBO energy
        qubo_energy = compiled_qubo.energy(bitstring)

        if not feas_result.is_feasible and status == "FEASIBLE":
            status = "INFEASIBLE"

        return SolverResult(
            solver_name=self.name,
            status=status,
            is_feasible=feas_result.is_feasible,
            objective_value=feas_result.objective_value,
            qubo_energy=qubo_energy,
            wall_time=wall_time,
            assignment=assignment,
            raw_bitstring=bitstring,
            num_violations=feas_result.num_violations,
            max_violation=feas_result.max_violation,
            solver_info=solver_info or {},
        )
