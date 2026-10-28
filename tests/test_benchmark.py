"""
Tests for benchmark problem generators and dataset runner.
"""
import pytest
from src.benchmark.generator import (
    generate_job_shop,
    generate_knapsack,
    generate_max_cut,
    generate_set_cover,
    generate_graph_coloring,
    build_instance_schedule,
    build_problem_from_spec,
)
from src.benchmark.runner import run_instance, DATASET_SOLVERS
from src.dsl.compiler import QUBOCompiler
from src.solvers.native_cpsat import NativeCPSATSolver


def test_generate_job_shop_structure():
    prob = generate_job_shop(n_jobs=3, n_slots=3, seed=0)
    assert prob.name.startswith("JobShop")
    assert "x" in prob.variables
    assert prob.variables["x"].size == 9  # 3 * 3
    assert len(prob.constraints) == 3 + 3  # 3 one-hot + 3 capacity
    assert prob.objective is not None


def test_generate_job_shop_feasible():
    prob = generate_job_shop(n_jobs=3, n_slots=3, seed=42)
    solver = NativeCPSATSolver()
    res = solver.solve(prob)
    assert res.is_feasible


def test_generate_knapsack_structure():
    prob = generate_knapsack(n_items=5, capacity_ratio=0.5, seed=0)
    assert "x" in prob.variables
    assert prob.variables["x"].size == 5
    assert len(prob.constraints) == 1  # weight constraint


def test_generate_knapsack_feasible():
    prob = generate_knapsack(n_items=8, capacity_ratio=0.6, seed=42)
    solver = NativeCPSATSolver()
    res = solver.solve(prob)
    assert res.is_feasible


def test_generate_max_cut_no_constraints():
    prob = generate_max_cut(n_nodes=6, edge_prob=0.5, seed=0)
    assert len(prob.constraints) == 0  # Pure QUBO, no hard constraints


def test_generate_max_cut_compiles():
    prob = generate_max_cut(n_nodes=8, edge_prob=0.4, seed=42)
    qubo = QUBOCompiler().compile(prob)
    assert qubo.num_orig_vars == 8
    assert qubo.num_slack_vars == 0


def test_generate_set_cover_feasible():
    prob = generate_set_cover(n_elements=4, n_sets=6, coverage_prob=0.5, seed=42)
    solver = NativeCPSATSolver()
    res = solver.solve(prob)
    assert res.is_feasible


def test_generate_graph_coloring_structure():
    prob = generate_graph_coloring(n_nodes=4, n_colors=4, edge_prob=0.4, seed=0)
    assert prob.variables["x"].size == 16  # 4 * 4
    # One-hot per vertex: 4 constraints. Edge coloring: varies.
    assert len(prob.constraints) >= 4


def test_generate_graph_coloring_feasible():
    # With n_colors == n_nodes, always feasible (assign unique color to each vertex)
    prob = generate_graph_coloring(n_nodes=4, n_colors=4, edge_prob=0.5, seed=42)
    solver = NativeCPSATSolver()
    res = solver.solve(prob)
    assert res.is_feasible


def test_build_instance_schedule_count():
    specs = build_instance_schedule(n_instances_per_type=10, rng_seed=1)
    assert len(specs) == 50  # 10 per type × 5 types
    types = [s.problem_type for s in specs]
    for t in ["job_shop", "knapsack", "max_cut", "set_cover", "graph_color"]:
        assert types.count(t) == 10


def test_job_shop_slots_gte_jobs():
    """Guarantee: all generated job shop instances have n_slots >= n_jobs."""
    specs = build_instance_schedule(n_instances_per_type=20, rng_seed=99)
    for spec in specs:
        if spec.problem_type == "job_shop":
            assert spec.params["n_slots"] >= spec.params["n_jobs"], (
                f"n_slots={spec.params['n_slots']} < n_jobs={spec.params['n_jobs']}"
            )


def test_run_instance_returns_row():
    prob = generate_job_shop(n_jobs=2, n_slots=2, seed=0)
    row = run_instance(instance_id=0, problem=prob, problem_type="job_shop")
    assert row is not None
    assert row.instance_id == 0
    assert row.problem_type == "job_shop"
    assert len(row.features) > 0
    assert set(DATASET_SOLVERS).issubset(set(row.solver_results.keys()))
    assert row.label in DATASET_SOLVERS + ["none"]
