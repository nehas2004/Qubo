"""
Parametric QUBO Problem Generator.

Generates OptimizationProblem AST instances for a variety of combinatorial
problem classes at configurable scales. Used to produce the ML training dataset.

Supported problem types:
  - job_shop    : N-Job M-TimeSlot scheduling (one-hot + machine capacity)
  - knapsack    : N-item binary knapsack (maximize value, weight constraint)
  - max_cut     : Max-Cut on a random graph
  - set_cover   : Minimum weighted set cover (greedy feasible instances)
  - graph_color : Graph k-coloring (vertex coloring with k colors)
"""

import random
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from src.dsl.ast_nodes import (
    OptimizationProblem,
    VariableDecl,
    VarType,
    VarRef,
    LinearTerm,
    QuadraticTerm,
    Expression,
    Objective,
    ObjectiveSense,
    Constraint,
    ComparisonOp,
)


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _var(name: str, idx: int) -> VarRef:
    return VarRef(name=name, index=idx)


def _linear(coeff: float, name: str, idx: int) -> LinearTerm:
    return LinearTerm(coeff=coeff, var=_var(name, idx))


def _quad(coeff: float, name: str, i: int, j: int) -> QuadraticTerm:
    return QuadraticTerm(coeff=coeff, var1=_var(name, i), var2=_var(name, j))


def _expr_linear(terms: List[Tuple[float, str, int]], constant: float = 0.0) -> Expression:
    return Expression(
        constant=constant,
        linear_terms=[_linear(c, n, i) for c, n, i in terms],
    )


def _constraint(name: str, lhs: Expression, op: ComparisonOp, rhs: float) -> Constraint:
    return Constraint(name=name, lhs=lhs, op=op, rhs=rhs)


# ---------------------------------------------------------------------------
# Problem Generators
# ---------------------------------------------------------------------------

def generate_job_shop(
    n_jobs: int,
    n_slots: int,
    seed: int = 0,
    name: Optional[str] = None,
) -> OptimizationProblem:
    """
    N-Job M-TimeSlot Job Shop scheduling problem.

    Variables: x[job * n_slots + slot] = 1 if job j starts at time t
    Objective: MINIMIZE sum (slot+1) * x[j*n_slots + slot]  (earlier is better)
    Constraints:
      - One-hot per job: sum_t x[j*n_slots + t] == 1 for each j
      - Capacity per slot: sum_j x[j*n_slots + t] <= 1 for each t
    """
    rng = random.Random(seed)
    n_vars = n_jobs * n_slots
    prob_name = name or f"JobShop_{n_jobs}j_{n_slots}t_s{seed}"

    problem = OptimizationProblem(name=prob_name)
    problem.variables["x"] = VariableDecl(name="x", var_type=VarType.BINARY, size=n_vars)

    # Objective: minimize sum of (slot_weight * x)
    # Add slight random perturbation to weights so instances aren't identical
    obj_terms = []
    for j in range(n_jobs):
        for t in range(n_slots):
            idx = j * n_slots + t
            weight = float(t + 1) + rng.uniform(0, 0.3)
            obj_terms.append(_linear(weight, "x", idx))

    problem.objective = Objective(
        sense=ObjectiveSense.MINIMIZE,
        expr=Expression(linear_terms=obj_terms),
    )

    # Constraint 1: One-hot per job
    for j in range(n_jobs):
        lhs = _expr_linear([(1.0, "x", j * n_slots + t) for t in range(n_slots)])
        problem.constraints.append(
            _constraint(f"job{j}_assigned", lhs, ComparisonOp.EQ, 1.0)
        )

    # Constraint 2: Capacity per time slot (at most 1 job per slot)
    for t in range(n_slots):
        lhs = _expr_linear([(1.0, "x", j * n_slots + t) for j in range(n_jobs)])
        problem.constraints.append(
            _constraint(f"slot{t}_capacity", lhs, ComparisonOp.LE, 1.0)
        )

    return problem


def generate_knapsack(
    n_items: int,
    capacity_ratio: float = 0.5,
    seed: int = 0,
    name: Optional[str] = None,
) -> OptimizationProblem:
    """
    Binary Knapsack problem.
    Variables: x[i] = 1 if item i is selected.
    Objective: MAXIMIZE sum(value[i] * x[i])
    Constraint: sum(weight[i] * x[i]) <= capacity
    """
    rng = random.Random(seed)
    prob_name = name or f"Knapsack_{n_items}i_s{seed}"

    values = [float(rng.randint(1, 20)) for _ in range(n_items)]
    weights = [float(rng.randint(1, 15)) for _ in range(n_items)]
    capacity = max(1.0, round(capacity_ratio * sum(weights)))

    problem = OptimizationProblem(name=prob_name)
    problem.variables["x"] = VariableDecl(name="x", var_type=VarType.BINARY, size=n_items)

    problem.objective = Objective(
        sense=ObjectiveSense.MAXIMIZE,
        expr=Expression(linear_terms=[_linear(values[i], "x", i) for i in range(n_items)]),
    )

    lhs = _expr_linear([(weights[i], "x", i) for i in range(n_items)])
    problem.constraints.append(
        _constraint("weight_limit", lhs, ComparisonOp.LE, capacity)
    )

    return problem


def generate_max_cut(
    n_nodes: int,
    edge_prob: float = 0.4,
    seed: int = 0,
    name: Optional[str] = None,
) -> OptimizationProblem:
    """
    Binary Max-Cut problem on a random Erdos-Renyi graph.
    Variables: x[i] in {0,1}, partition assignment for node i.
    QUBO objective: MAXIMIZE sum_{(i,j) in E} (x[i] + x[j] - 2*x[i]*x[j])
    (cut = 1 when nodes are in different partitions)
    """
    rng = random.Random(seed)
    prob_name = name or f"MaxCut_{n_nodes}n_s{seed}"

    edges: List[Tuple[int, int]] = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                edges.append((i, j))

    problem = OptimizationProblem(name=prob_name)
    problem.variables["x"] = VariableDecl(name="x", var_type=VarType.BINARY, size=n_nodes)

    linear_terms = []
    quad_terms = []
    for (i, j) in edges:
        linear_terms.append(_linear(1.0, "x", i))
        linear_terms.append(_linear(1.0, "x", j))
        quad_terms.append(_quad(-2.0, "x", i, j))

    problem.objective = Objective(
        sense=ObjectiveSense.MAXIMIZE,
        expr=Expression(linear_terms=linear_terms, quadratic_terms=quad_terms),
    )

    # No hard constraints: pure unconstrained QUBO
    return problem


def generate_set_cover(
    n_elements: int,
    n_sets: int,
    coverage_prob: float = 0.4,
    seed: int = 0,
    name: Optional[str] = None,
) -> OptimizationProblem:
    """
    Minimum Weighted Set Cover problem.
    Variables: x[s] = 1 if set s is selected.
    Objective: MINIMIZE sum(cost[s] * x[s])
    Constraints: For each element e, at least one set covering e must be selected.
    """
    rng = random.Random(seed)
    prob_name = name or f"SetCover_{n_elements}e_{n_sets}s_s{seed}"

    # coverage[s][e] = 1 if set s covers element e
    coverage: List[List[bool]] = [
        [rng.random() < coverage_prob for _ in range(n_elements)]
        for _ in range(n_sets)
    ]

    # Ensure every element is covered by at least one set
    for e in range(n_elements):
        if not any(coverage[s][e] for s in range(n_sets)):
            s = rng.randint(0, n_sets - 1)
            coverage[s][e] = True

    costs = [float(rng.randint(1, 10)) for _ in range(n_sets)]

    problem = OptimizationProblem(name=prob_name)
    problem.variables["x"] = VariableDecl(name="x", var_type=VarType.BINARY, size=n_sets)

    problem.objective = Objective(
        sense=ObjectiveSense.MINIMIZE,
        expr=Expression(linear_terms=[_linear(costs[s], "x", s) for s in range(n_sets)]),
    )

    for e in range(n_elements):
        covering_sets = [s for s in range(n_sets) if coverage[s][e]]
        if covering_sets:
            lhs = _expr_linear([(1.0, "x", s) for s in covering_sets])
            problem.constraints.append(
                _constraint(f"cover_elem{e}", lhs, ComparisonOp.GE, 1.0)
            )

    return problem


def generate_graph_coloring(
    n_nodes: int,
    n_colors: int,
    edge_prob: float = 0.4,
    seed: int = 0,
    name: Optional[str] = None,
) -> OptimizationProblem:
    """
    Graph k-coloring feasibility/minimization problem.
    Variables: x[v * n_colors + c] = 1 if vertex v gets color c.
    Objective: MINIMIZE total coloring cost (uniform = feasibility focus).
    Constraints:
      - One color per vertex (one-hot)
      - No adjacent vertices share a color (edge coloring constraint)
    """
    rng = random.Random(seed)
    prob_name = name or f"GraphColor_{n_nodes}n_{n_colors}c_s{seed}"

    edges: List[Tuple[int, int]] = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                edges.append((i, j))

    n_vars = n_nodes * n_colors
    problem = OptimizationProblem(name=prob_name)
    problem.variables["x"] = VariableDecl(name="x", var_type=VarType.BINARY, size=n_vars)

    # Objective: minimize total variables used (feasibility-oriented)
    problem.objective = Objective(
        sense=ObjectiveSense.MINIMIZE,
        expr=Expression(
            linear_terms=[_linear(1.0, "x", i) for i in range(n_vars)],
        ),
    )

    # Constraint 1: One-hot coloring per vertex
    for v in range(n_nodes):
        lhs = _expr_linear([(1.0, "x", v * n_colors + c) for c in range(n_colors)])
        problem.constraints.append(
            _constraint(f"vertex{v}_color", lhs, ComparisonOp.EQ, 1.0)
        )

    # Constraint 2: No adjacent vertices share color
    for (u, v) in edges:
        for c in range(n_colors):
            lhs = _expr_linear([
                (1.0, "x", u * n_colors + c),
                (1.0, "x", v * n_colors + c),
            ])
            problem.constraints.append(
                _constraint(f"edge_{u}_{v}_color{c}", lhs, ComparisonOp.LE, 1.0)
            )

    return problem


# ---------------------------------------------------------------------------
# Instance Schedule: defines what instances to generate for the full dataset
# ---------------------------------------------------------------------------

@dataclass
class InstanceSpec:
    problem_type: str
    params: dict
    seed: int


def build_instance_schedule(
    n_instances_per_type: int = 80,
    rng_seed: int = 42,
) -> List[InstanceSpec]:
    """
    Returns a schedule of problem instances to generate for the full dataset.
    Covers all 5 problem types across a range of sizes and seeds.
    """
    rng = random.Random(rng_seed)
    specs: List[InstanceSpec] = []

    seeds = [rng.randint(0, 99999) for _ in range(n_instances_per_type)]

    # Job Shop: 2-7 jobs × n_jobs to (n_jobs+4) time slots
    # Guarantee n_slots >= n_jobs. Larger instances favour heuristic solvers.
    for i, seed in enumerate(seeds):
        n_jobs = rng.randint(2, 7)
        n_slots = rng.randint(n_jobs, n_jobs + 4)
        specs.append(InstanceSpec(
            problem_type="job_shop",
            params={"n_jobs": n_jobs, "n_slots": n_slots},
            seed=seed,
        ))

    # Knapsack: 4-30 items, varying capacity ratios
    # Larger instances favour SA/Tabu over CP-SAT
    for i, seed in enumerate(seeds):
        n_items = rng.randint(4, 30)
        cap_ratio = round(rng.uniform(0.2, 0.8), 2)
        specs.append(InstanceSpec(
            problem_type="knapsack",
            params={"n_items": n_items, "capacity_ratio": cap_ratio},
            seed=seed,
        ))

    # Max Cut: 5-20 nodes, varying edge densities
    # Dense Max-Cut is an ideal benchmark for quantum-inspired solvers
    for i, seed in enumerate(seeds):
        n_nodes = rng.randint(5, 20)
        edge_prob = round(rng.uniform(0.3, 0.8), 2)
        specs.append(InstanceSpec(
            problem_type="max_cut",
            params={"n_nodes": n_nodes, "edge_prob": edge_prob},
            seed=seed,
        ))

    # Set Cover: 4-12 elements, 5-18 sets
    for i, seed in enumerate(seeds):
        n_elems = rng.randint(4, 12)
        n_sets = rng.randint(5, 18)
        cov_prob = round(rng.uniform(0.25, 0.55), 2)
        specs.append(InstanceSpec(
            problem_type="set_cover",
            params={"n_elements": n_elems, "n_sets": n_sets, "coverage_prob": cov_prob},
            seed=seed,
        ))

    # Graph Coloring: 3-7 nodes, always use n_colors >= ceil(n_nodes/2) so instances are feasible
    for i, seed in enumerate(seeds):
        n_nodes = rng.randint(3, 7)
        # Guarantee enough colors: use at least ceil(n_nodes/2) to ensure chromatic feasibility
        n_colors = rng.randint(max(2, (n_nodes + 1) // 2), n_nodes)
        edge_prob = round(rng.uniform(0.2, 0.5), 2)
        specs.append(InstanceSpec(
            problem_type="graph_color",
            params={"n_nodes": n_nodes, "n_colors": n_colors, "edge_prob": edge_prob},
            seed=seed,
        ))

    return specs


def build_problem_from_spec(spec: InstanceSpec) -> OptimizationProblem:
    """Instantiates an OptimizationProblem from an InstanceSpec."""
    generators = {
        "job_shop": generate_job_shop,
        "knapsack": generate_knapsack,
        "max_cut": generate_max_cut,
        "set_cover": generate_set_cover,
        "graph_color": generate_graph_coloring,
    }
    gen_fn = generators[spec.problem_type]
    return gen_fn(**spec.params, seed=spec.seed)
