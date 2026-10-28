"""
Tests for Classical, Quantum-Inspired, and Quantum Hardware solvers.
"""

import pytest

from src.dsl.parser import DSLParser
from src.solvers import (
    get_solver,
    list_solvers,
    benchmark_solvers,
    SimulatedAnnealingSolver,
    TabuSolver,
    SimulatedQuantumAnnealingSolver,
    SimulatedBifurcationSolver,
    DWaveQPUSolver,
    CPSATUnifiedSolver,
)


@pytest.fixture
def toy_scheduling_problem():
    parser = DSLParser()
    return parser.parse_file("examples/toy_scheduling.opt")


def test_list_solvers():
    solvers = list_solvers()
    assert "cpsat" in solvers
    assert "sa" in solvers
    assert "tabu" in solvers
    assert "sqa" in solvers
    assert "sbm" in solvers
    assert "qpu" in solvers


def test_cpsat_unified_solver(toy_scheduling_problem):
    solver = CPSATUnifiedSolver()
    result = solver.solve(toy_scheduling_problem)

    assert result.solver_name == "NativeCP-SAT"
    assert result.status == "OPTIMAL"
    assert result.is_feasible is True
    assert result.objective_value == 3.0
    # Optimal scheduling: x[0]=1, x[3]=1 (early start for job 1, delayed start for job 2 to prevent overlap)
    # or x[1]=1, x[2]=1 -> objective is 1*1 + 2*1 = 3.0
    assert result.assignment.get("x[0]") in (0.0, 1.0)


def test_simulated_annealing_solver(toy_scheduling_problem):
    solver = SimulatedAnnealingSolver(num_reads=50, num_sweeps=500, seed=42)
    result = solver.solve(toy_scheduling_problem)

    assert result.solver_name == "SimulatedAnnealing"
    assert result.wall_time > 0
    assert result.raw_bitstring is not None
    assert result.is_feasible is True
    assert result.objective_value == 3.0


def test_tabu_solver(toy_scheduling_problem):
    solver = TabuSolver(num_reads=30, timeout=500, seed=42)
    result = solver.solve(toy_scheduling_problem)

    assert result.solver_name == "TabuSearch"
    assert result.wall_time > 0
    assert result.raw_bitstring is not None
    assert result.is_feasible is True
    assert result.objective_value == 3.0


def test_simulated_quantum_annealing_solver(toy_scheduling_problem):
    solver = SimulatedQuantumAnnealingSolver(
        num_trotter_slices=8, num_steps=200, beta=6.0, seed=42
    )
    result = solver.solve(toy_scheduling_problem)

    assert result.solver_name == "SimulatedQuantumAnnealing"
    assert result.wall_time > 0
    assert result.raw_bitstring is not None
    assert result.is_feasible is True
    assert result.objective_value == 3.0


def test_simulated_bifurcation_solver(toy_scheduling_problem):
    solver = SimulatedBifurcationSolver(num_steps=300, num_shots=30, dt=0.5, seed=42)
    result = solver.solve(toy_scheduling_problem)

    assert result.solver_name == "SimulatedBifurcation"
    assert result.wall_time > 0
    assert result.raw_bitstring is not None
    assert result.is_feasible is True
    assert result.objective_value == 3.0


def test_dwave_qpu_no_token_graceful(toy_scheduling_problem):
    solver = DWaveQPUSolver(token=None)
    result = solver.solve(toy_scheduling_problem)

    assert result.solver_name == "DWaveQPU"
    assert result.status == "ERROR"
    assert result.is_feasible is False
    assert "error" in result.solver_info or "exception" in result.solver_info


def test_benchmark_solvers_portfolio(toy_scheduling_problem):
    results = benchmark_solvers(
        toy_scheduling_problem, solvers=["cpsat", "sa", "tabu", "sqa", "sbm"]
    )
    assert len(results) == 5
    for res in results:
        assert res.wall_time >= 0
        assert res.is_feasible is True
        assert res.objective_value in (3.0, 4.0)
