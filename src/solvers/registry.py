"""
Central registry, factory, and benchmarking harness for all QUBO and classical solvers.
"""

from typing import Dict, List, Type, Optional, Any
from src.dsl.ast_nodes import OptimizationProblem
from src.dsl.compiler import CompiledQUBO, QUBOCompiler
from .base import BaseQUBOSolver, SolverResult
from .native_cpsat import NativeCPSATSolver, CPSATUnifiedSolver
from .simulated_annealing import SimulatedAnnealingSolver
from .tabu_search import TabuSolver
from .simulated_quantum_annealing import SimulatedQuantumAnnealingSolver
from .simulated_bifurcation import SimulatedBifurcationSolver
from .dwave_qpu import DWaveQPUSolver


SOLVER_REGISTRY: Dict[str, Type[BaseQUBOSolver]] = {
    "cpsat": CPSATUnifiedSolver,
    "sa": SimulatedAnnealingSolver,
    "simulated_annealing": SimulatedAnnealingSolver,
    "tabu": TabuSolver,
    "tabu_search": TabuSolver,
    "sqa": SimulatedQuantumAnnealingSolver,
    "simulated_quantum_annealing": SimulatedQuantumAnnealingSolver,
    "sbm": SimulatedBifurcationSolver,
    "simulated_bifurcation": SimulatedBifurcationSolver,
    "dwave": DWaveQPUSolver,
    "qpu": DWaveQPUSolver,
}

SOLVER_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "cpsat": {
        "category": "Classical Exact",
        "description": "Google OR-Tools CP-SAT (lazy clause generation SAT/CP). Provable global optimum.",
    },
    "sa": {
        "category": "Classical Heuristic",
        "description": "dwave-neal Simulated Annealing (thermal MCMC with geometric cooling).",
    },
    "tabu": {
        "category": "Classical Local Search",
        "description": "dwave-tabu Tabu Search (neighborhood exploration with adaptive memory).",
    },
    "sqa": {
        "category": "Quantum-Inspired",
        "description": "Path-Integral Monte Carlo Simulated Quantum Annealing (Trotter quantum tunneling).",
    },
    "sbm": {
        "category": "Quantum-Inspired",
        "description": "Ballistic Simulated Bifurcation Machine (Kerr parametric oscillator network).",
    },
    "qpu": {
        "category": "Quantum Hardware",
        "description": "D-Wave Advantage Quantum Processing Unit (superconducting flux qubit annealer).",
    },
}


def list_solvers() -> Dict[str, Dict[str, str]]:
    """Returns a dictionary of all canonical solver names and their descriptions."""
    return SOLVER_DESCRIPTIONS


def get_solver(name: str, **kwargs) -> BaseQUBOSolver:
    """Instantiates a solver by key name."""
    clean_name = name.lower().strip()
    if clean_name not in SOLVER_REGISTRY:
        available = list(SOLVER_DESCRIPTIONS.keys())
        raise ValueError(f"Unknown solver '{name}'. Available solvers: {available}")

    solver_cls = SOLVER_REGISTRY[clean_name]
    return solver_cls(**kwargs)


def benchmark_solvers(
    problem: OptimizationProblem,
    solvers: Optional[List[str]] = None,
    penalty: Optional[float] = None,
    **solver_kwargs,
) -> List[SolverResult]:
    """
    Runs a portfolio of solvers against a single OptimizationProblem instance
    and compiles a comparative performance report.
    """
    if solvers is None:
        # Default portfolio of local solvers (excluding real cloud hardware)
        solvers = ["cpsat", "sa", "tabu", "sqa", "sbm"]

    compiler = QUBOCompiler(default_penalty=penalty)
    compiled_qubo = compiler.compile(problem)

    results: List[SolverResult] = []
    for s_name in solvers:
        try:
            solver = get_solver(s_name)
            res = solver.solve(problem, compiled_qubo=compiled_qubo, **solver_kwargs)
            results.append(res)
        except Exception as e:
            results.append(
                SolverResult(
                    solver_name=s_name,
                    status="ERROR",
                    is_feasible=False,
                    objective_value=None,
                    qubo_energy=None,
                    wall_time=0.0,
                    assignment={},
                    num_violations=len(problem.constraints),
                    solver_info={"exception": str(e)},
                )
            )

    return results
