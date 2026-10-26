"""
AST (Abstract Syntax Tree) nodes representing optimization problems.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Union


class VarType(Enum):
    BINARY = "binary"
    INTEGER = "integer"


class ObjectiveSense(Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ComparisonOp(Enum):
    EQ = "=="
    LE = "<="
    GE = ">="


@dataclass
class VariableDecl:
    """Declaration of a variable or 1D variable array."""
    name: str
    var_type: VarType = VarType.BINARY
    size: Optional[int] = None  # None for scalar, integer N for array of size N
    lower_bound: int = 0
    upper_bound: int = 1


@dataclass
class VarRef:
    """Reference to a variable, either scalar (name) or indexed (name, index)."""
    name: str
    index: Optional[int] = None

    def to_string(self) -> str:
        if self.index is not None:
            return f"{self.name}[{self.index}]"
        return self.name

    def __hash__(self):
        return hash((self.name, self.index))

    def __eq__(self, other):
        if not isinstance(other, VarRef):
            return False
        return self.name == other.name and self.index == other.index


@dataclass
class LinearTerm:
    """A linear term: coeff * var."""
    coeff: float
    var: VarRef


@dataclass
class QuadraticTerm:
    """A quadratic term: coeff * var1 * var2."""
    coeff: float
    var1: VarRef
    var2: VarRef


@dataclass
class Expression:
    """
    Polynomial expression up to degree 2:
    constant + sum(linear_terms) + sum(quadratic_terms)
    """
    constant: float = 0.0
    linear_terms: List[LinearTerm] = field(default_factory=list)
    quadratic_terms: List[QuadraticTerm] = field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            self.constant == 0.0
            and len(self.linear_terms) == 0
            and len(self.quadratic_terms) == 0
        )


@dataclass
class Objective:
    """Objective function specification."""
    sense: ObjectiveSense
    expr: Expression


@dataclass
class Constraint:
    """A constraint: expr op rhs, e.g., sum(x[i]) == 1."""
    name: str
    lhs: Expression
    op: ComparisonOp
    rhs: float = 0.0


@dataclass
class OptimizationProblem:
    """Complete AST for a parsed optimization problem."""
    name: str = "Problem"
    variables: Dict[str, VariableDecl] = field(default_factory=dict)
    objective: Optional[Objective] = None
    constraints: List[Constraint] = field(default_factory=list)

    def get_all_var_refs(self) -> List[VarRef]:
        """Returns all individual variable references declared in the problem."""
        refs = []
        for name, decl in self.variables.items():
            if decl.size is not None:
                for idx in range(decl.size):
                    refs.append(VarRef(name=name, index=idx))
            else:
                refs.append(VarRef(name=name, index=None))
        return refs
