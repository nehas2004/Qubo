"""
Compiler module lowering AST OptimizationProblem into:
1. Upper-triangular QUBO matrix Q (NumPy 2D array or dict)
2. Set B structural metadata features
3. Native CP-SAT formulation model
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
import networkx as nx

from .ast_nodes import (
    OptimizationProblem,
    ObjectiveSense,
    ComparisonOp,
    VarRef,
    LinearTerm,
    QuadraticTerm,
    Expression,
)


@dataclass
class ConstraintPenaltySpec:
    """Penalty specification for a lowered constraint in QUBO space."""
    name: str
    terms: List[Tuple[int, float]]  # List of (matrix_index, coefficient)
    rhs: float


@dataclass
class CompiledQUBO:
    """Result of compiling an optimization problem to QUBO."""
    matrix: np.ndarray                   # N x N upper-triangular matrix
    var_map: Dict[str, int]              # Variable display string -> matrix index
    index_to_var: Dict[int, str]         # Matrix index -> Variable display string
    constant_offset: float               # Constant energy shift
    set_b_features: Dict[str, float]     # Set B features extracted during compilation
    num_orig_vars: int
    num_slack_vars: int
    penalty_multiplier: float
    constraint_specs: List[ConstraintPenaltySpec] = field(default_factory=list)

    def energy(self, bitstring: List[int]) -> float:
        """Evaluate the objective value / energy for a given binary solution."""
        x = np.array(bitstring, dtype=float)
        return float(x @ self.matrix @ x + self.constant_offset)

    def constraint_violation_energy(self, bitstring: List[int]) -> float:
        """Calculates total penalty energy incurred by constraints."""
        b = np.array(bitstring, dtype=float)
        total = 0.0
        for spec in self.constraint_specs:
            lhs_val = sum(c * b[idx] for idx, c in spec.terms)
            viol = lhs_val - spec.rhs
            total += self.penalty_multiplier * (viol ** 2)
        return float(total)

    def is_qubo_state_consistent(self, bitstring: List[int], tolerance: float = 1e-6) -> bool:
        """Checks if extended state has zero constraint penalty (slack matches)."""
        return self.constraint_violation_energy(bitstring) <= tolerance

    def to_dict(self) -> Dict[Tuple[int, int], float]:
        """Convert matrix to coordinate dictionary {(i, j): coeff} for D-Wave / dimod."""
        res = {}
        n = self.matrix.shape[0]
        for i in range(n):
            for j in range(i, n):
                val = self.matrix[i, j]
                if abs(val) > 1e-12:
                    res[(i, j)] = float(val)
        return res


class QUBOCompiler:
    """Compiles OptimizationProblem AST into QUBO matrix and extracts Set B features."""

    def __init__(self, default_penalty: Optional[float] = None):
        """
        :param default_penalty: Optional explicit penalty multiplier. If None, computes adaptively.
        """
        self.default_penalty = default_penalty

    def compile(self, problem: OptimizationProblem) -> CompiledQUBO:
        """Perform the compilation pipeline."""
        # 1. Gather all original variables and assign indices
        orig_var_refs = problem.get_all_var_refs()
        var_to_idx: Dict[str, int] = {}
        idx_to_var: Dict[int, str] = {}
        for idx, ref in enumerate(orig_var_refs):
            s = ref.to_string()
            var_to_idx[s] = idx
            idx_to_var[idx] = s

        num_orig_vars = len(var_to_idx)
        current_var_count = num_orig_vars

        # Dictionaries accumulating linear and quadratic coefficients
        # Key: (i, j) with i <= j
        qubo_coeffs: Dict[Tuple[int, int], float] = defaultdict(float)
        constant_offset = 0.0

        # 2. Compile Objective
        # Standard QUBO is MINIMIZATION: min x^T Q x
        obj_sense_mult = 1.0
        if problem.objective:
            if problem.objective.sense == ObjectiveSense.MAXIMIZE:
                obj_sense_mult = -1.0

            constant_offset += obj_sense_mult * problem.objective.expr.constant

            for lt in problem.objective.expr.linear_terms:
                i = var_to_idx[lt.var.to_string()]
                qubo_coeffs[(i, i)] += obj_sense_mult * lt.coeff

            for qt in problem.objective.expr.quadratic_terms:
                i = var_to_idx[qt.var1.to_string()]
                j = var_to_idx[qt.var2.to_string()]
                pair = (min(i, j), max(i, j))
                qubo_coeffs[pair] += obj_sense_mult * qt.coeff

        # Calculate scale of objective coefficients to determine adaptive penalty
        obj_abs_coeffs = [abs(c) for c in qubo_coeffs.values() if abs(c) > 1e-12]
        max_obj_coeff = max(obj_abs_coeffs) if obj_abs_coeffs else 1.0
        mean_obj_coeff = float(np.mean(obj_abs_coeffs)) if obj_abs_coeffs else 1.0

        if self.default_penalty is not None:
            penalty = float(self.default_penalty)
        else:
            # Adaptive rule: penalty must exceed maximum conceivable objective gain
            penalty = max(10.0, 2.0 * sum(obj_abs_coeffs) + 1.0)

        # 3. Compile Constraints
        count_eq = 0
        count_ineq = 0
        slack_var_counter = 0
        constraint_specs: List[ConstraintPenaltySpec] = []

        for con in problem.constraints:
            if con.op == ComparisonOp.EQ:
                count_eq += 1
                # Constraint: sum(a_i * x_i) == b
                # Penalty: P * (sum(a_i * x_i) - b)^2
                terms: List[Tuple[int, float]] = []
                for lt in con.lhs.linear_terms:
                    idx = var_to_idx[lt.var.to_string()]
                    terms.append((idx, lt.coeff))
                b = con.rhs
                constraint_specs.append(ConstraintPenaltySpec(name=con.name, terms=list(terms), rhs=float(b)))
                constant_offset += self._add_quadratic_penalty(
                    terms=terms,
                    rhs=b,
                    penalty=penalty,
                    qubo_coeffs=qubo_coeffs,
                )

            elif con.op in (ComparisonOp.LE, ComparisonOp.GE):
                count_ineq += 1
                # Normalize GE to LE: sum(a_i * x_i) >= b  ==>  sum(-a_i * x_i) <= -b
                terms: List[Tuple[int, float]] = []
                for lt in con.lhs.linear_terms:
                    idx = var_to_idx[lt.var.to_string()]
                    coeff = lt.coeff if con.op == ComparisonOp.LE else -lt.coeff
                    terms.append((idx, coeff))
                b = con.rhs if con.op == ComparisonOp.LE else -con.rhs

                # Estimate max possible slack needed:
                # slack = b - sum(a_i * x_i). Max slack occurs when sum(a_i * x_i) is minimized.
                min_lhs = sum(c for _, c in terms if c < 0.0)
                max_slack_val = max(1, math.ceil(b - min_lhs))

                # Binary expansion of slack: s = sum(2^k * s_k)
                # Introduce binary slack variables
                power = 1
                remaining_slack = max_slack_val
                while remaining_slack > 0:
                    weight = min(power, remaining_slack)
                    slack_name = f"_slack_{slack_var_counter}"
                    slack_idx = current_var_count
                    var_to_idx[slack_name] = slack_idx
                    idx_to_var[slack_idx] = slack_name
                    current_var_count += 1
                    slack_var_counter += 1

                    terms.append((slack_idx, float(weight)))
                    remaining_slack -= weight
                    power *= 2

                # Now add penalty: P * (sum(a_i * x_i) + sum(w_k * s_k) - b)^2
                constraint_specs.append(ConstraintPenaltySpec(name=con.name, terms=list(terms), rhs=float(b)))
                constant_offset += self._add_quadratic_penalty(
                    terms=terms,
                    rhs=b,
                    penalty=penalty,
                    qubo_coeffs=qubo_coeffs,
                )

        total_vars = current_var_count
        num_slack_vars = slack_var_counter

        # 4. Construct N x N Matrix
        matrix = np.zeros((total_vars, total_vars), dtype=float)
        for (i, j), val in qubo_coeffs.items():
            if abs(val) > 1e-12:
                matrix[min(i, j), max(i, j)] += val

        # 5. Extract Set B Features
        # Compute graph densities
        g_native = nx.Graph()
        g_native.add_nodes_from(range(num_orig_vars))
        for (i, j), val in qubo_coeffs.items():
            if i < num_orig_vars and j < num_orig_vars and i != j and abs(val) > 1e-12:
                g_native.add_edge(i, j)

        g_qubo = nx.Graph()
        g_qubo.add_nodes_from(range(total_vars))
        for (i, j), val in qubo_coeffs.items():
            if i != j and abs(val) > 1e-12:
                g_qubo.add_edge(i, j)

        density_native = nx.density(g_native) if num_orig_vars > 1 else 0.0
        density_qubo = nx.density(g_qubo) if total_vars > 1 else 0.0

        degrees = [d for _, d in g_qubo.degree()]
        mean_degree = float(np.mean(degrees)) if degrees else 0.0
        max_degree = float(np.max(degrees)) if degrees else 0.0

        set_b = {
            "b_num_orig_vars": float(num_orig_vars),
            "b_num_slack_vars": float(num_slack_vars),
            "b_total_vars": float(total_vars),
            "b_slack_ratio": float(num_slack_vars / total_vars) if total_vars > 0 else 0.0,
            "b_num_constraints": float(count_eq + count_ineq),
            "b_num_eq_constraints": float(count_eq),
            "b_num_ineq_constraints": float(count_ineq),
            "b_constraints_per_var": float((count_eq + count_ineq) / num_orig_vars) if num_orig_vars > 0 else 0.0,
            "b_penalty_multiplier": float(penalty),
            "b_penalty_to_obj_max_ratio": float(penalty / max_obj_coeff),
            "b_penalty_to_obj_mean_ratio": float(penalty / mean_obj_coeff),
            "b_graph_density_native": float(density_native),
            "b_graph_density_qubo": float(density_qubo),
            "b_graph_mean_degree": float(mean_degree),
            "b_graph_max_degree": float(max_degree),
        }

        return CompiledQUBO(
            matrix=matrix,
            var_map=var_to_idx,
            index_to_var=idx_to_var,
            constant_offset=constant_offset,
            set_b_features=set_b,
            num_orig_vars=num_orig_vars,
            num_slack_vars=num_slack_vars,
            penalty_multiplier=penalty,
            constraint_specs=constraint_specs,
        )

    def _add_quadratic_penalty(
        self,
        terms: List[Tuple[int, float]],
        rhs: float,
        penalty: float,
        qubo_coeffs: Dict[Tuple[int, int], float],
    ) -> float:
        """
        Adds P * (sum(c_i * x_i) - b)^2 to qubo_coeffs.
        Expansion:
        (sum c_i x_i - b)^2 = (sum c_i x_i)^2 - 2b sum(c_i x_i) + b^2
        = sum c_i^2 x_i^2 + 2 sum_{i < j} c_i c_j x_i x_j - 2b sum c_i x_i + b^2
        Since x_i^2 = x_i for binary variables:
        = sum (c_i^2 - 2b c_i) x_i + 2 sum_{i < j} c_i c_j x_i x_j + b^2
        Returns the constant offset: P * b^2.
        """
        # Linear terms: P * (c_i^2 - 2 * rhs * c_i)
        for idx, coeff in terms:
            qubo_coeffs[(idx, idx)] += penalty * (coeff ** 2 - 2.0 * rhs * coeff)

        # Cross quadratic terms: P * 2 * c_i * c_j
        n_terms = len(terms)
        for k1 in range(n_terms):
            idx1, c1 = terms[k1]
            for k2 in range(k1 + 1, n_terms):
                idx2, c2 = terms[k2]
                pair = (min(idx1, idx2), max(idx1, idx2))
                qubo_coeffs[pair] += penalty * (2.0 * c1 * c2)

        return penalty * (rhs ** 2)
