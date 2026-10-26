"""
Parser implementation converting DSL source text into an OptimizationProblem AST.
"""

from typing import Dict, Tuple, Optional, List
from collections import defaultdict
from lark import Lark, Transformer, v_args, Token

from .ast_nodes import (
    VarType,
    ObjectiveSense,
    ComparisonOp,
    VariableDecl,
    VarRef,
    LinearTerm,
    QuadraticTerm,
    Expression,
    Objective,
    Constraint,
    OptimizationProblem,
)
from .grammar import DSL_GRAMMAR


def _canonical_pair(v1: VarRef, v2: VarRef) -> Tuple[VarRef, VarRef]:
    """Return a deterministic ordered pair of variables."""
    s1 = (v1.name, -1 if v1.index is None else v1.index)
    s2 = (v2.name, -1 if v2.index is None else v2.index)
    if s1 <= s2:
        return (v1, v2)
    return (v2, v1)


class Polynomial:
    """
    Represents an algebraic polynomial up to degree 2:
    constant + sum(a_i * x_i) + sum(q_ij * x_i * x_j)
    """

    def __init__(
        self,
        constant: float = 0.0,
        linear: Optional[Dict[VarRef, float]] = None,
        quadratic: Optional[Dict[Tuple[VarRef, VarRef], float]] = None,
    ):
        self.constant: float = float(constant)
        self.linear: Dict[VarRef, float] = defaultdict(float)
        if linear:
            for k, v in linear.items():
                if abs(v) > 1e-12:
                    self.linear[k] += v

        self.quadratic: Dict[Tuple[VarRef, VarRef], float] = defaultdict(float)
        if quadratic:
            for (v1, v2), c in quadratic.items():
                if abs(c) > 1e-12:
                    pair = _canonical_pair(v1, v2)
                    self.quadratic[pair] += c

    def __add__(self, other: "Polynomial") -> "Polynomial":
        res = Polynomial(self.constant + other.constant)
        for v, c in self.linear.items():
            res.linear[v] += c
        for v, c in other.linear.items():
            res.linear[v] += c
        for p, c in self.quadratic.items():
            res.quadratic[p] += c
        for p, c in other.quadratic.items():
            res.quadratic[p] += c
        res._clean()
        return res

    def __sub__(self, other: "Polynomial") -> "Polynomial":
        res = Polynomial(self.constant - other.constant)
        for v, c in self.linear.items():
            res.linear[v] += c
        for v, c in other.linear.items():
            res.linear[v] -= c
        for p, c in self.quadratic.items():
            res.quadratic[p] += c
        for p, c in other.quadratic.items():
            res.quadratic[p] -= c
        res._clean()
        return res

    def __neg__(self) -> "Polynomial":
        res = Polynomial(-self.constant)
        for v, c in self.linear.items():
            res.linear[v] = -c
        for p, c in self.quadratic.items():
            res.quadratic[p] = -c
        return res

    def __mul__(self, other: "Polynomial") -> "Polynomial":
        # Check if multiplying these would result in degree > 2
        # self degree:
        has_quad1 = len(self.quadratic) > 0
        has_lin1 = len(self.linear) > 0
        has_quad2 = len(other.quadratic) > 0
        has_lin2 = len(other.linear) > 0

        if (has_quad1 and (has_lin2 or has_quad2)) or (has_quad2 and (has_lin1 or has_quad1)):
            raise ValueError("QUBO expressions cannot exceed polynomial degree 2 (quadratic).")

        res = Polynomial(self.constant * other.constant)

        # c1 * lin2
        if self.constant != 0.0:
            for v, c in other.linear.items():
                res.linear[v] += self.constant * c
            for p, c in other.quadratic.items():
                res.quadratic[p] += self.constant * c

        # c2 * lin1
        if other.constant != 0.0:
            for v, c in self.linear.items():
                res.linear[v] += other.constant * c
            for p, c in self.quadratic.items():
                res.quadratic[p] += other.constant * c

        # lin1 * lin2
        for v1, c1 in self.linear.items():
            for v2, c2 in other.linear.items():
                if v1 == v2:
                    # In binary optimization, x_i * x_i == x_i
                    res.linear[v1] += c1 * c2
                else:
                    pair = _canonical_pair(v1, v2)
                    res.quadratic[pair] += c1 * c2

        res._clean()
        return res

    def _clean(self):
        self.linear = defaultdict(float, {k: v for k, v in self.linear.items() if abs(v) > 1e-12})
        self.quadratic = defaultdict(float, {k: v for k, v in self.quadratic.items() if abs(v) > 1e-12})

    def to_expression(self) -> Expression:
        self._clean()
        lin = [LinearTerm(coeff=c, var=v) for v, c in self.linear.items()]
        quad = [QuadraticTerm(coeff=c, var1=p[0], var2=p[1]) for p, c in self.quadratic.items()]
        return Expression(constant=self.constant, linear_terms=lin, quadratic_terms=quad)


class _DSLTransformer(Transformer):
    def problem_name(self, children) -> str:
        return str(children[0])

    def var_decl_binary_array(self, children) -> VariableDecl:
        name = str(children[0])
        size = int(children[1])
        return VariableDecl(name=name, var_type=VarType.BINARY, size=size)

    def var_decl_binary_scalar(self, children) -> VariableDecl:
        name = str(children[0])
        return VariableDecl(name=name, var_type=VarType.BINARY, size=None)

    def var_decl_int_array(self, children) -> VariableDecl:
        name = str(children[0])
        size = int(children[1])
        lb = int(children[2])
        ub = int(children[3])
        return VariableDecl(name=name, var_type=VarType.INTEGER, size=size, lower_bound=lb, upper_bound=ub)

    def var_decl_int_scalar(self, children) -> VariableDecl:
        name = str(children[0])
        lb = int(children[1])
        ub = int(children[2])
        return VariableDecl(name=name, var_type=VarType.INTEGER, size=None, lower_bound=lb, upper_bound=ub)

    def var_section(self, children) -> Dict[str, VariableDecl]:
        return {decl.name: decl for decl in children}

    def OBJ_SENSE(self, token: Token) -> ObjectiveSense:
        if str(token).lower() == "minimize":
            return ObjectiveSense.MINIMIZE
        return ObjectiveSense.MAXIMIZE

    def obj_section(self, children) -> Objective:
        sense = children[0]
        poly: Polynomial = children[1]
        return Objective(sense=sense, expr=poly.to_expression())

    def COMP_OP(self, token: Token) -> ComparisonOp:
        return ComparisonOp(str(token))

    def constraint_def(self, children) -> Constraint:
        # Children could be: [name, lhs_poly, op, rhs_poly] OR [lhs_poly, op, rhs_poly]
        if len(children) == 4:
            name = str(children[0])
            lhs_poly = children[1]
            op = children[2]
            rhs_poly = children[3]
        else:
            name = "c"
            lhs_poly = children[0]
            op = children[1]
            rhs_poly = children[2]

        # Normalize constraint so that lhs is an Expression and rhs is a scalar constant
        # e.g., expr1 <= expr2  -->  (expr1 - expr2).lhs_poly op 0.0
        diff_poly: Polynomial = lhs_poly - rhs_poly
        # Move constant to RHS: poly - const op -const
        rhs_val = -diff_poly.constant
        diff_poly.constant = 0.0
        return Constraint(name=name, lhs=diff_poly.to_expression(), op=op, rhs=rhs_val)

    def con_section(self, children) -> List[Constraint]:
        res = []
        for idx, con in enumerate(children):
            if con.name == "c":
                con.name = f"c_{idx}"
            res.append(con)
        return res

    def problem(self, children) -> OptimizationProblem:
        name = "Problem"
        if children[0] is not None:
            name = str(children[0])

        var_dict = children[1]
        obj = children[2]
        constraints = children[3] if len(children) > 3 and children[3] is not None else []

        return OptimizationProblem(
            name=name,
            variables=var_dict,
            objective=obj,
            constraints=constraints,
        )

    # Expression parsing rules
    def num(self, children) -> Polynomial:
        return Polynomial(constant=float(children[0]))

    def var_indexed(self, children) -> Polynomial:
        v = VarRef(name=str(children[0]), index=int(children[1]))
        return Polynomial(linear={v: 1.0})

    def var_scalar(self, children) -> Polynomial:
        v = VarRef(name=str(children[0]), index=None)
        return Polynomial(linear={v: 1.0})

    def term(self, children) -> Polynomial:
        res = children[0]
        i = 1
        while i < len(children):
            # children[i] is MUL_OP '*'
            next_factor = children[i + 1]
            res = res * next_factor
            i += 2
        return res

    def expr(self, children) -> Polynomial:
        idx = 0
        first_sign = 1.0
        if children[0] is None:
            idx += 1
        elif isinstance(children[0], Token) and children[0].type == "SIGN":
            if children[0].value == "-":
                first_sign = -1.0
            idx += 1

        res = children[idx]
        if first_sign == -1.0:
            res = -res
        idx += 1

        while idx < len(children):
            op = str(children[idx])
            next_term = children[idx + 1]
            if op == "+":
                res = res + next_term
            elif op == "-":
                res = res - next_term
            idx += 2
        return res


class DSLParser:
    """Parser class with a compiled Lark grammar."""

    def __init__(self):
        self._lark = Lark(DSL_GRAMMAR, start="problem", parser="lalr")
        self._transformer = _DSLTransformer()

    def parse(self, text: str) -> OptimizationProblem:
        """Parse source text into an OptimizationProblem AST."""
        tree = self._lark.parse(text)
        problem: OptimizationProblem = self._transformer.transform(tree)
        self._validate(problem)
        return problem

    def parse_file(self, filepath: str) -> OptimizationProblem:
        """Read and parse a .opt file."""
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return self.parse(text)

    def _validate(self, problem: OptimizationProblem):
        """Validate variable references within the problem."""
        declared_vars = problem.variables

        def check_var(v: VarRef):
            if v.name not in declared_vars:
                raise ValueError(f"Variable '{v.name}' referenced but not declared.")
            decl = declared_vars[v.name]
            if decl.size is not None:
                if v.index is None:
                    raise ValueError(f"Array variable '{v.name}' referenced without an index.")
                if v.index < 0 or v.index >= decl.size:
                    raise ValueError(
                        f"Index {v.index} out of bounds for array '{v.name}' (size: {decl.size})."
                    )
            else:
                if v.index is not None:
                    raise ValueError(f"Scalar variable '{v.name}' referenced with an index [{v.index}].")

        # Validate objective
        if problem.objective:
            for lt in problem.objective.expr.linear_terms:
                check_var(lt.var)
            for qt in problem.objective.expr.quadratic_terms:
                check_var(qt.var1)
                check_var(qt.var2)

        # Validate constraints
        for con in problem.constraints:
            for lt in con.lhs.linear_terms:
                check_var(lt.var)
            for qt in con.lhs.quadratic_terms:
                check_var(qt.var1)
                check_var(qt.var2)
