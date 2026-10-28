"""
Native CP-SAT solver backend for Optimization DSL AST.

Directly translates an OptimizationProblem AST into an OR-Tools CP-SAT model
without lowering to QUBO penalties, providing the exact ground-truth solution.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
from ortools.sat.python import cp_model

from src.dsl.ast_nodes import (
    OptimizationProblem,
    VarType,
    ObjectiveSense,
    ComparisonOp,
    Expression,
    LinearTerm,
    QuadraticTerm,
)


from src.dsl.checker import check_solution_feasibility
from .base import BaseQUBOSolver, SolverResult


@dataclass
class NativeCPSATResult:
    """Result of solving a native model with CP-SAT."""
    status: str                         # "OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNKNOWN"
    is_feasible: bool
    objective_value: Optional[float]
    best_bound: Optional[float]
    wall_time: float
    assignment: Dict[str, float] = field(default_factory=dict)
    solver_stats: str = ""

    def to_solver_result(self, problem: Optional[OptimizationProblem] = None) -> SolverResult:
        """Converts native result to standardized SolverResult."""
        violations = 0
        max_viol = 0.0
        if problem and self.assignment:
            feas = check_solution_feasibility(problem, self.assignment)
            violations = feas.num_violations
            max_viol = feas.max_violation

        return SolverResult(
            solver_name="NativeCP-SAT",
            status=self.status,
            is_feasible=self.is_feasible,
            objective_value=self.objective_value,
            qubo_energy=None,
            wall_time=self.wall_time,
            assignment=self.assignment,
            num_violations=violations,
            max_violation=max_viol,
            solver_info={"best_bound": self.best_bound, "stats": self.solver_stats},
        )


class NativeCPSATSolver:
    """Translates OptimizationProblem to CpModel and solves natively."""

    def __init__(self, precision_scale: int = 1000):
        """
        :param precision_scale: Integer multiplier to preserve fractional coefficients in CP-SAT.
        """
        self.precision_scale = precision_scale

    def solve(
        self,
        problem: OptimizationProblem,
        time_limit_sec: Optional[float] = None,
        random_seed: int = 42,
    ) -> NativeCPSATResult:
        """Translates problem to CpModel and executes solver."""
        model = cp_model.CpModel()

        # 1. Determine scaling factor needed for fractional coefficients
        scale = self._compute_scaling_factor(problem)

        # 2. Declare variables
        cp_vars: Dict[str, Any] = {}
        for name, decl in problem.variables.items():
            if decl.size is not None:
                for idx in range(decl.size):
                    ref_str = f"{name}[{idx}]"
                    if decl.var_type == VarType.BINARY:
                        cp_vars[ref_str] = model.new_bool_var(ref_str)
                    else:
                        cp_vars[ref_str] = model.new_int_var(
                            decl.lower_bound, decl.upper_bound, ref_str
                        )
            else:
                if decl.var_type == VarType.BINARY:
                    cp_vars[name] = model.new_bool_var(name)
                else:
                    cp_vars[name] = model.new_int_var(
                        decl.lower_bound, decl.upper_bound, name
                    )

        # Helper to linearize quadratic terms via product variables
        prod_vars: Dict[Tuple[str, str], Any] = {}

        def get_product_var(s1: str, s2: str) -> Any:
            pair = (min(s1, s2), max(s1, s2))
            if pair in prod_vars:
                return prod_vars[pair]

            v1 = cp_vars[pair[0]]
            v2 = cp_vars[pair[1]]
            p_name = f"_prod_{pair[0]}_{pair[1]}"
            p_var = model.new_bool_var(p_name)
            model.add_multiplication_equality(p_var, [v1, v2])
            prod_vars[pair] = p_var
            return p_var

        # 3. Add Constraints
        for con in problem.constraints:
            expr_terms = []
            for lt in con.lhs.linear_terms:
                v = cp_vars[lt.var.to_string()]
                c_int = int(round(lt.coeff * scale))
                expr_terms.append(c_int * v)

            for qt in con.lhs.quadratic_terms:
                s1 = qt.var1.to_string()
                s2 = qt.var2.to_string()
                p_var = get_product_var(s1, s2)
                c_int = int(round(qt.coeff * scale))
                expr_terms.append(c_int * p_var)

            lhs_sum = sum(expr_terms)
            # Subtract constant if lhs had one
            rhs_int = int(round((con.rhs - con.lhs.constant) * scale))

            if con.op == ComparisonOp.EQ:
                model.add(lhs_sum == rhs_int)
            elif con.op == ComparisonOp.LE:
                model.add(lhs_sum <= rhs_int)
            elif con.op == ComparisonOp.GE:
                model.add(lhs_sum >= rhs_int)

        # 4. Set Objective
        if problem.objective:
            obj_terms = []
            for lt in problem.objective.expr.linear_terms:
                v = cp_vars[lt.var.to_string()]
                c_int = int(round(lt.coeff * scale))
                obj_terms.append(c_int * v)

            for qt in problem.objective.expr.quadratic_terms:
                s1 = qt.var1.to_string()
                s2 = qt.var2.to_string()
                p_var = get_product_var(s1, s2)
                c_int = int(round(qt.coeff * scale))
                obj_terms.append(c_int * p_var)

            obj_sum = sum(obj_terms)
            const_int = int(round(problem.objective.expr.constant * scale))
            full_obj = obj_sum + const_int

            if problem.objective.sense == ObjectiveSense.MINIMIZE:
                model.minimize(full_obj)
            else:
                model.maximize(full_obj)

        # 5. Solve
        solver = cp_model.CpSolver()
        if time_limit_sec is not None:
            solver.parameters.max_time_in_seconds = float(time_limit_sec)
        solver.parameters.random_seed = int(random_seed)

        status_code = solver.solve(model)
        status_name = solver.status_name(status_code)

        is_feasible = status_name in ("OPTIMAL", "FEASIBLE")
        obj_val: Optional[float] = None
        bound_val: Optional[float] = None
        assignment: Dict[str, float] = {}

        if is_feasible:
            raw_obj = solver.objective_value
            obj_val = float(raw_obj / scale) if problem.objective else 0.0
            try:
                bound_val = float(solver.best_objective_bound / scale)
            except Exception:
                bound_val = obj_val

            for k, v in cp_vars.items():
                assignment[k] = float(solver.value(v))

        return NativeCPSATResult(
            status=status_name,
            is_feasible=is_feasible,
            objective_value=obj_val,
            best_bound=bound_val,
            wall_time=float(solver.wall_time),
            assignment=assignment,
            solver_stats=solver.response_stats(),
        )

    def _compute_scaling_factor(self, problem: OptimizationProblem) -> int:
        """Determines if coefficients are fractional and returns integer scaling."""
        all_coeffs = []
        if problem.objective:
            all_coeffs.append(problem.objective.expr.constant)
            all_coeffs.extend(lt.coeff for lt in problem.objective.expr.linear_terms)
            all_coeffs.extend(qt.coeff for qt in problem.objective.expr.quadratic_terms)

        for con in problem.constraints:
            all_coeffs.append(con.rhs)
            all_coeffs.append(con.lhs.constant)
            all_coeffs.extend(lt.coeff for lt in con.lhs.linear_terms)
            all_coeffs.extend(qt.coeff for qt in con.lhs.quadratic_terms)

        has_fraction = any(not float(c).is_integer() for c in all_coeffs if abs(c) > 1e-9)
        return self.precision_scale if has_fraction else 1


class CPSATUnifiedSolver(BaseQUBOSolver):
    """
    Adapter exposing NativeCPSATSolver under the BaseQUBOSolver unified interface.
    """

    def __init__(self, precision_scale: int = 1000):
        super().__init__(name="NativeCP-SAT")
        self.underlying_solver = NativeCPSATSolver(precision_scale=precision_scale)

    def solve(
        self,
        problem: OptimizationProblem,
        compiled_qubo: Optional[Any] = None,
        time_limit_sec: Optional[float] = None,
        random_seed: int = 42,
        **kwargs,
    ) -> SolverResult:
        res = self.underlying_solver.solve(
            problem, time_limit_sec=time_limit_sec, random_seed=random_seed
        )
        return res.to_solver_result(problem=problem)
