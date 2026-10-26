"""
Language Semantic and Feasibility Checker for Optimization DSL.

Provides:
1. Semantic AST validation (domain bounds, undeclared variables, index bounds).
2. Solution feasibility checking (evaluating candidate assignments against native constraints).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .ast_nodes import (
    OptimizationProblem,
    VariableDecl,
    VarRef,
    VarType,
    ObjectiveSense,
    ComparisonOp,
    Expression,
    Constraint,
)


@dataclass
class ConstraintViolation:
    """Details of a single constraint violation."""
    constraint_name: str
    op: ComparisonOp
    lhs_value: float
    rhs_value: float
    violation_magnitude: float  # Absolute violation (e.g., > 0 if violated, 0 if satisfied)


@dataclass
class FeasibilityResult:
    """Result of evaluating a solution against problem constraints."""
    is_feasible: bool
    objective_value: Optional[float]
    violations: List[ConstraintViolation] = field(default_factory=list)
    max_violation: float = 0.0
    num_violations: int = 0

    def summary(self) -> str:
        if self.is_feasible:
            return f"FEASIBLE (Objective: {self.objective_value:.4f})" if self.objective_value is not None else "FEASIBLE"
        return f"INFEASIBLE ({self.num_violations} violations, max violation: {self.max_violation:.4f})"


def validate_semantics(problem: OptimizationProblem) -> List[str]:
    """
    Validates the semantic consistency of an OptimizationProblem AST.
    Returns a list of error messages (empty if valid).
    """
    errors: List[str] = []
    declared_vars = problem.variables

    if not declared_vars:
        errors.append("Model defines no variables.")

    # Validate variable declarations
    for name, decl in declared_vars.items():
        if decl.size is not None and decl.size <= 0:
            errors.append(f"Variable array '{name}' has invalid non-positive size: {decl.size}.")
        if decl.var_type == VarType.INTEGER and decl.lower_bound > decl.upper_bound:
            errors.append(
                f"Integer variable '{name}' has lower_bound ({decl.lower_bound}) > upper_bound ({decl.upper_bound})."
            )

    def check_var_ref(v: VarRef, context: str):
        if v.name not in declared_vars:
            errors.append(f"In {context}: Variable '{v.name}' is referenced but not declared.")
            return
        decl = declared_vars[v.name]
        if decl.size is not None:
            if v.index is None:
                errors.append(f"In {context}: Array variable '{v.name}' referenced without an index.")
            elif v.index < 0 or v.index >= decl.size:
                errors.append(
                    f"In {context}: Index {v.index} out of bounds for array '{v.name}' (declared size: {decl.size})."
                )
        else:
            if v.index is not None:
                errors.append(f"In {context}: Scalar variable '{v.name}' referenced with index [{v.index}].")

    # Validate objective
    if problem.objective:
        for lt in problem.objective.expr.linear_terms:
            check_var_ref(lt.var, "OBJECTIVE")
        for qt in problem.objective.expr.quadratic_terms:
            check_var_ref(qt.var1, "OBJECTIVE")
            check_var_ref(qt.var2, "OBJECTIVE")

    # Validate constraints
    constraint_names = set()
    for con in problem.constraints:
        if con.name in constraint_names:
            errors.append(f"Duplicate constraint name: '{con.name}'.")
        constraint_names.add(con.name)

        for lt in con.lhs.linear_terms:
            check_var_ref(lt.var, f"CONSTRAINT '{con.name}'")
        for qt in con.lhs.quadratic_terms:
            check_var_ref(qt.var1, f"CONSTRAINT '{con.name}'")
            check_var_ref(qt.var2, f"CONSTRAINT '{con.name}'")

    return errors


def evaluate_expression(expr: Expression, assignment: Dict[str, float]) -> float:
    """Evaluates an AST polynomial Expression given variable values."""
    val = float(expr.constant)

    for lt in expr.linear_terms:
        s = lt.var.to_string()
        v_val = assignment.get(s, 0.0)
        val += lt.coeff * v_val

    for qt in expr.quadratic_terms:
        s1 = qt.var1.to_string()
        s2 = qt.var2.to_string()
        v1_val = assignment.get(s1, 0.0)
        v2_val = assignment.get(s2, 0.0)
        val += qt.coeff * v1_val * v2_val

    return float(val)


def check_solution_feasibility(
    problem: OptimizationProblem,
    assignment: Dict[str, float],
    tolerance: float = 1e-6,
) -> FeasibilityResult:
    """
    Checks whether a concrete assignment satisfies all problem constraints
    and evaluates the objective function.
    """
    violations: List[ConstraintViolation] = []

    # Check bounds on declared variables
    for name, decl in problem.variables.items():
        refs = []
        if decl.size is not None:
            refs = [f"{name}[{i}]" for i in range(decl.size)]
        else:
            refs = [name]

        for r in refs:
            if r in assignment:
                val = assignment[r]
                lb = 0.0 if decl.var_type == VarType.BINARY else float(decl.lower_bound)
                ub = 1.0 if decl.var_type == VarType.BINARY else float(decl.upper_bound)
                if val < lb - tolerance or val > ub + tolerance:
                    violations.append(
                        ConstraintViolation(
                            constraint_name=f"bound_{r}",
                            op=ComparisonOp.LE,
                            lhs_value=val,
                            rhs_value=ub if val > ub else lb,
                            violation_magnitude=max(val - ub, lb - val),
                        )
                    )

    # Check constraints
    for con in problem.constraints:
        lhs_val = evaluate_expression(con.lhs, assignment)
        rhs_val = float(con.rhs)
        diff = lhs_val - rhs_val

        viol_mag = 0.0
        if con.op == ComparisonOp.EQ:
            if abs(diff) > tolerance:
                viol_mag = abs(diff)
        elif con.op == ComparisonOp.LE:
            if diff > tolerance:
                viol_mag = diff
        elif con.op == ComparisonOp.GE:
            if -diff > tolerance:
                viol_mag = -diff

        if viol_mag > 0.0:
            violations.append(
                ConstraintViolation(
                    constraint_name=con.name,
                    op=con.op,
                    lhs_value=lhs_val,
                    rhs_value=rhs_val,
                    violation_magnitude=viol_mag,
                )
            )

    is_feasible = len(violations) == 0
    max_viol = max((v.violation_magnitude for v in violations), default=0.0)

    # Calculate objective value
    obj_val: Optional[float] = None
    if problem.objective:
        obj_val = evaluate_expression(problem.objective.expr, assignment)

    return FeasibilityResult(
        is_feasible=is_feasible,
        objective_value=obj_val,
        violations=violations,
        max_violation=max_viol,
        num_violations=len(violations),
    )


def bitstring_to_assignment(
    bitstring: List[int],
    index_to_var: Dict[int, str],
) -> Dict[str, float]:
    """Converts a bitstring and index mapping to a variable assignment dictionary."""
    return {index_to_var[idx]: float(val) for idx, val in enumerate(bitstring) if idx in index_to_var}
