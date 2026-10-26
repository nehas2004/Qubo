"""
Tests for semantic validation and solution feasibility checker.
"""

import pytest

from src.dsl.parser import DSLParser
from src.dsl.ast_nodes import OptimizationProblem, VariableDecl, VarType, VarRef, LinearTerm, Expression, Constraint, ComparisonOp
from src.dsl.checker import validate_semantics, check_solution_feasibility, evaluate_expression


def test_semantic_validation_valid():
    code = """
    PROBLEM: ValidProb
    VARIABLES:
        binary x[3]
        binary y
        integer z in 0..5
    OBJECTIVE:
        MINIMIZE 2*x[0] + y - z
    CONSTRAINTS:
        c1: x[0] + x[1] <= 1
        c2: y + z == 2
    """
    prob = DSLParser().parse(code)
    errors = validate_semantics(prob)
    assert len(errors) == 0


def test_semantic_validation_errors():
    parser = DSLParser()

    # Model with empty variables
    prob = OptimizationProblem(name="Empty")
    errors = validate_semantics(prob)
    assert any("no variables" in e for e in errors)

    # Model with invalid bounds
    prob.variables["k"] = VariableDecl(name="k", var_type=VarType.INTEGER, lower_bound=5, upper_bound=2)
    errors = validate_semantics(prob)
    assert any("lower_bound" in e for e in errors)

    # Model referencing undeclared variable in constraint
    prob.variables["x"] = VariableDecl(name="x", var_type=VarType.BINARY, size=2)
    con = Constraint(
        name="c1",
        lhs=Expression(linear_terms=[LinearTerm(coeff=1.0, var=VarRef(name="undeclared"))]),
        op=ComparisonOp.EQ,
        rhs=1.0,
    )
    prob.constraints.append(con)
    errors = validate_semantics(prob)
    assert any("undeclared" in e for e in errors)

    # Model referencing out of bounds index
    con2 = Constraint(
        name="c2",
        lhs=Expression(linear_terms=[LinearTerm(coeff=1.0, var=VarRef(name="x", index=5))]),
        op=ComparisonOp.EQ,
        rhs=1.0,
    )
    prob.constraints.append(con2)
    errors = validate_semantics(prob)
    assert any("Index 5 out of bounds" in e for e in errors)


def test_solution_feasibility_checker():
    code = """
    PROBLEM: FeasTest
    VARIABLES:
        binary x[3]
    OBJECTIVE:
        MINIMIZE 10*x[0] + 20*x[1] + 30*x[2]
    CONSTRAINTS:
        one_hot: x[0] + x[1] + x[2] == 1
        cap: x[0] + x[1] <= 1
    """
    prob = DSLParser().parse(code)

    # Feasible solution: x[0]=1, x[1]=0, x[2]=0
    sol_feasible = {"x[0]": 1.0, "x[1]": 0.0, "x[2]": 0.0}
    res = check_solution_feasibility(prob, sol_feasible)
    assert res.is_feasible is True
    assert res.num_violations == 0
    assert res.objective_value == 10.0

    # Infeasible: both one_hot and cap violated (sum = 2)
    sol_infeas_sum = {"x[0]": 1.0, "x[1]": 1.0, "x[2]": 0.0}
    res2 = check_solution_feasibility(prob, sol_infeas_sum)
    assert res2.is_feasible is False
    assert res2.num_violations == 2
    assert res2.violations[0].constraint_name == "one_hot"
    assert res2.violations[0].violation_magnitude == 1.0

    # Infeasible: bounds violated (x[0] = 2.0)
    sol_infeas_bounds = {"x[0]": 2.0, "x[1]": 0.0, "x[2]": 0.0}
    res3 = check_solution_feasibility(prob, sol_infeas_bounds)
    assert res3.is_feasible is False
    assert any("bound_x[0]" in v.constraint_name for v in res3.violations)
