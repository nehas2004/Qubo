"""
Unit tests for the DSL Lexer, Parser, QUBO Compiler, and Feature Extractors.
"""

import pytest
import numpy as np

from src.dsl.parser import DSLParser, Polynomial
from src.dsl.ast_nodes import VarRef, VarType, ObjectiveSense, ComparisonOp
from src.dsl.compiler import QUBOCompiler
from src.features.set_a import extract_set_a_features


def test_polynomial_algebra():
    x0 = VarRef("x", 0)
    x1 = VarRef("x", 1)

    p1 = Polynomial(linear={x0: 2.0})
    p2 = Polynomial(linear={x1: 3.0})

    # Addition
    psum = p1 + p2
    assert psum.linear[x0] == 2.0
    assert psum.linear[x1] == 3.0

    # Multiplication (Cross term)
    pprod = p1 * p2
    pair = (x0, x1) if (x0.name, x0.index) <= (x1.name, x1.index) else (x1, x0)
    assert pprod.quadratic[pair] == 6.0

    # Idempotence: x0 * x0 == x0 for binary variables
    psq = p1 * p1
    assert psq.linear[x0] == 4.0
    assert len(psq.quadratic) == 0

    # Degree > 2 rejection
    p_quad = Polynomial(quadratic={pair: 1.0})
    with pytest.raises(ValueError, match="degree 2"):
        _ = p_quad * p1


def test_parser_basic_model():
    code = """
    PROBLEM: SimpleTest
    VARIABLES:
        binary x[3]
    OBJECTIVE:
        MINIMIZE 2*x[0] - 3*x[1] + 4*x[0]*x[1]
    CONSTRAINTS:
        c1: x[0] + x[1] + x[2] == 1
    """
    parser = DSLParser()
    prob = parser.parse(code)

    assert prob.name == "SimpleTest"
    assert "x" in prob.variables
    assert prob.variables["x"].size == 3
    assert prob.objective.sense == ObjectiveSense.MINIMIZE
    assert len(prob.constraints) == 1
    assert prob.constraints[0].name == "c1"
    assert prob.constraints[0].op == ComparisonOp.EQ
    assert prob.constraints[0].rhs == 1.0


def test_compiler_equality_penalty():
    code = """
    VARIABLES:
        binary x[2]
    OBJECTIVE:
        MINIMIZE 0
    CONSTRAINTS:
        one_hot: x[0] + x[1] == 1
    """
    parser = DSLParser()
    prob = parser.parse(code)

    # Use fixed penalty P = 10.0
    compiler = QUBOCompiler(default_penalty=10.0)
    compiled = compiler.compile(prob)

    # (x0 + x1 - 1)^2 = x0^2 + x1^2 + 2 x0 x1 - 2 x0 - 2 x1 + 1
    # Since x^2 = x:
    # = (1 - 2) x0 + (1 - 2) x1 + 2 x0 x1 + 1
    # = -1 x0 - 1 x1 + 2 x0 x1 + 1
    # Multiplied by P=10:
    # Q_00 = -10, Q_11 = -10, Q_01 = 20, constant_offset = 10
    Q = compiled.matrix
    assert Q[0, 0] == -10.0
    assert Q[1, 1] == -10.0
    assert Q[0, 1] == 20.0
    assert compiled.constant_offset == 10.0

    # Test energy of solutions:
    # [1, 0] -> satisfies constraint -> energy should be 0
    assert compiled.energy([1, 0]) == 0.0
    # [0, 1] -> satisfies constraint -> energy should be 0
    assert compiled.energy([0, 1]) == 0.0
    # [0, 0] -> violates constraint (gap=1) -> penalty = 10
    assert compiled.energy([0, 0]) == 10.0
    # [1, 1] -> violates constraint (gap=1) -> penalty = 10
    assert compiled.energy([1, 1]) == 10.0


def test_compiler_inequality_slack():
    code = """
    VARIABLES:
        binary x[2]
    OBJECTIVE:
        MINIMIZE 0
    CONSTRAINTS:
        capacity: x[0] + x[1] <= 1
    """
    parser = DSLParser()
    prob = parser.parse(code)

    compiler = QUBOCompiler(default_penalty=10.0)
    compiled = compiler.compile(prob)

    # Must have introduced 1 slack variable: total 3 vars
    assert compiled.num_orig_vars == 2
    assert compiled.num_slack_vars == 1
    assert compiled.matrix.shape == (3, 3)

    # Valid solution [0, 1, 0] (x0=0, x1=1, slack=0) -> energy = 0
    assert compiled.energy([0, 1, 0]) == 0.0
    # Valid solution [0, 0, 1] (x0=0, x1=0, slack=1) -> energy = 0
    assert compiled.energy([0, 0, 1]) == 0.0


def test_feature_extraction():
    # Construct a known 3x3 QUBO
    Q = np.array([
        [2.0, 3.0, 0.0],
        [0.0, -1.0, 4.0],
        [0.0, 0.0, 5.0]
    ])
    features = extract_set_a_features(Q)
    assert features["a_num_vars"] == 3.0
    assert features["a_num_linear_nonzero"] == 3.0
    assert features["a_num_quadratic_nonzero"] == 2.0
    # linear terms: 2, -1, 5 -> mean = 2.0
    assert np.isclose(features["a_mean_linear"], 2.0)
