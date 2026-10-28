"""
Solvers package for QUBO and native discrete optimization models.
"""

from .base import BaseQUBOSolver, SolverResult
from .native_cpsat import NativeCPSATSolver, NativeCPSATResult, CPSATUnifiedSolver
from .simulated_annealing import SimulatedAnnealingSolver
from .tabu_search import TabuSolver
from .simulated_quantum_annealing import SimulatedQuantumAnnealingSolver
from .simulated_bifurcation import SimulatedBifurcationSolver
from .dwave_qpu import DWaveQPUSolver
from .registry import get_solver, list_solvers, benchmark_solvers, SOLVER_REGISTRY

__all__ = [
    "BaseQUBOSolver",
    "SolverResult",
    "NativeCPSATSolver",
    "NativeCPSATResult",
    "CPSATUnifiedSolver",
    "SimulatedAnnealingSolver",
    "TabuSolver",
    "SimulatedQuantumAnnealingSolver",
    "SimulatedBifurcationSolver",
    "DWaveQPUSolver",
    "get_solver",
    "list_solvers",
    "benchmark_solvers",
    "SOLVER_REGISTRY",
]
