"""
Tests for Native CP-SAT Solver and QUBO Brute-Force Verifier.
"""

import pytest

from src.dsl.parser import DSLParser
from src.dsl.verifier import QUBOVerifier
from src.solvers.native_cpsat import NativeCPSATSolver


def test_native_cpsat_simple_knapsack():
    code = """
    PROBLEM: Knapsack
    VARIABLES:
        binary x[3]
    OBJECTIVE:
        MAXIMIZE 5*x[0] + 3*x[1] + 2*x[2]
    CONSTRAINTS:
        weight: 3*x[0] + 2*x[1] + 1*x[2] <= 4
    """
    prob = DSLParser().parse(code)
    solver = NativeCPSATSolver()
    res = solver.solve(prob)

    assert res.status == "OPTIMAL"
    assert res.is_feasible is True
    # Optimum: take item 0 (w=3, v=5) and item 2 (w=1, v=2) -> total weight 4, total value 7
    assert res.objective_value == 7.0
    assert res.assignment["x[0]"] == 1.0
    assert res.assignment["x[1]"] == 0.0
    assert res.assignment["x[2]"] == 1.0


def test_native_cpsat_infeasible():
    code = """
    PROBLEM: Impossible
    VARIABLES:
        binary x[1]
    OBJECTIVE:
        MINIMIZE x[0]
    CONSTRAINTS:
        c1: x[0] == 1
        c2: x[0] == 0
    """
    prob = DSLParser().parse(code)
    solver = NativeCPSATSolver()
    res = solver.solve(prob)

    assert res.status == "INFEASIBLE"
    assert res.is_feasible is False


def test_qubo_verifier_toy_scheduling():
    parser = DSLParser()
    prob = parser.parse_file("examples/toy_scheduling.opt")

    verifier = QUBOVerifier(max_brute_force_vars=20)
    report = verifier.verify(prob)

    assert report.is_valid is True
    assert report.native_status == "OPTIMAL"
    assert report.native_opt_obj == 3.0
    assert report.qubo_ground_energy == 3.0
    assert report.ground_state_feasible is True
    assert report.spectral_gap is not None and report.spectral_gap > 0.0
    assert report.exhaustively_verified is True


def test_qubo_verifier_one_hot_selection():
    code = """
    PROBLEM: OneHotMin
    VARIABLES:
        binary x[3]
    OBJECTIVE:
        MINIMIZE 10*x[0] + 4*x[1] + 7*x[2]
    CONSTRAINTS:
        exactly_one: x[0] + x[1] + x[2] == 1
    """
    prob = DSLParser().parse(code)

    verifier = QUBOVerifier(max_brute_force_vars=10)
    report = verifier.verify(prob)

    assert report.is_valid is True
    assert report.native_opt_obj == 4.0
    assert report.qubo_ground_energy == 4.0
    assert report.ground_state_feasible is True
    assert report.num_ground_states == 1
    assert report.spectral_gap > 0.0


def test_qubo_verifier_inequality_knapsack():
    code = """
    PROBLEM: SmallKnapsack
    VARIABLES:
        binary x[3]
    OBJECTIVE:
        MINIMIZE -3*x[0] - 2*x[1] - 1*x[2]
    CONSTRAINTS:
        cap: 2*x[0] + 2*x[1] + 1*x[2] <= 3
    """
    prob = DSLParser().parse(code)

    verifier = QUBOVerifier(max_brute_force_vars=10)
    report = verifier.verify(prob)

    assert report.is_valid is True
    # Take x[0]=1 (w=2, v=-3) and x[2]=1 (w=1, v=-1) -> obj = -4.0
    assert report.native_opt_obj == -4.0
    assert report.qubo_ground_energy == -4.0
    assert report.ground_state_feasible is True
    assert report.spectral_gap > 0.0
