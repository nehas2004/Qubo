"""
Dataset Runner: Runs all solvers on generated problem instances,
extracts features, records performance metrics, and writes dataset.csv.
"""

import time
import traceback
import csv
import os
import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.dsl.ast_nodes import OptimizationProblem
from src.dsl.compiler import QUBOCompiler, CompiledQUBO
from src.features.set_a import extract_set_a_features
from src.solvers import (
    benchmark_solvers,
    get_solver,
    SolverResult,
)


# Solvers to compare in dataset generation (excluding QPU which needs hardware)
DATASET_SOLVERS = ["cpsat", "sa", "tabu", "sqa", "sbm"]

# Features that will be used as ML inputs
FEATURE_KEYS_A = [
    "a_num_vars",
    "a_num_linear_nonzero",
    "a_num_quadratic_nonzero",
    "a_mean_linear",
    "a_var_linear",
    "a_mean_quadratic",
    "a_var_quadratic",
    "a_mean_all",
    "a_var_all",
]

FEATURE_KEYS_B = [
    "b_num_orig_vars",
    "b_num_slack_vars",
    "b_total_vars",
    "b_slack_ratio",
    "b_num_constraints",
    "b_num_eq_constraints",
    "b_num_ineq_constraints",
    "b_constraints_per_var",
    "b_penalty_multiplier",
    "b_penalty_to_obj_max_ratio",
    "b_penalty_to_obj_mean_ratio",
    "b_graph_density_native",
    "b_graph_density_qubo",
    "b_graph_mean_degree",
    "b_graph_max_degree",
]

ALL_FEATURE_KEYS = FEATURE_KEYS_A + FEATURE_KEYS_B

# Per-solver performance columns written to CSV
def _solver_columns(solver_name: str) -> List[str]:
    return [
        f"{solver_name}_status",
        f"{solver_name}_feasible",
        f"{solver_name}_objective",
        f"{solver_name}_qubo_energy",
        f"{solver_name}_time_ms",
        f"{solver_name}_violations",
    ]


ALL_SOLVER_COLUMNS = []
for s in DATASET_SOLVERS:
    ALL_SOLVER_COLUMNS.extend(_solver_columns(s))


@dataclass
class DatasetRow:
    """A single row in the generated dataset."""
    instance_id: int
    problem_name: str
    problem_type: str
    features: Dict[str, float] = field(default_factory=dict)
    solver_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    best_solver_by_time: str = ""        # Fastest solver that found feasible solution
    best_solver_by_quality: str = ""     # Solver with lowest objective among feasible
    best_solver_combined: str = ""       # Combined (time × gap) score winner
    label: str = ""                      # Primary ML label (best_solver_combined)


def _compute_label(
    solver_results: Dict[str, Dict[str, Any]],
    cpsat_obj: Optional[float],
) -> tuple:
    """
    Derives the 3 best-solver labels from raw solver performance data.
    """
    feasible_solvers = {
        s: r for s, r in solver_results.items()
        if r.get("feasible", False)
    }

    if not feasible_solvers:
        return ("none", "none", "none")

    # Best by time: fastest feasible solver
    best_time = min(feasible_solvers, key=lambda s: feasible_solvers[s]["time_ms"])

    # Best by quality: lowest objective value
    best_quality = min(
        feasible_solvers,
        key=lambda s: feasible_solvers[s].get("objective", float("inf")) or float("inf"),
    )

    # Combined score: time_ms × (1 + optimality_gap)
    # Optimality gap = (obj - opt_obj) / abs(opt_obj) if cpsat found opt
    best_combined = best_time
    best_combined_score = float("inf")

    for s_name, r in feasible_solvers.items():
        t = r["time_ms"]
        obj = r.get("objective")
        if obj is None:
            continue
        if cpsat_obj is not None and abs(cpsat_obj) > 1e-9:
            gap = max(0.0, (obj - cpsat_obj) / abs(cpsat_obj))
        else:
            gap = 0.0
        score = t * (1.0 + gap)
        if score < best_combined_score:
            best_combined_score = score
            best_combined = s_name

    return (best_time, best_quality, best_combined)


def run_instance(
    instance_id: int,
    problem: OptimizationProblem,
    problem_type: str,
    penalty: Optional[float] = None,
    time_limit_cpsat: float = 10.0,
    verbose: bool = False,
) -> Optional[DatasetRow]:
    """
    Runs all solvers on a single problem instance and returns a DatasetRow.
    Returns None if compilation fails (e.g., empty or degenerate problem).
    """
    try:
        # Step 1: Compile to QUBO
        compiler = QUBOCompiler(default_penalty=penalty)
        qubo: CompiledQUBO = compiler.compile(problem)

        if qubo.matrix.shape[0] == 0:
            return None

        # Step 2: Extract features
        set_a = extract_set_a_features(qubo.matrix)
        set_b = qubo.set_b_features
        features = {**{k: set_a.get(k, 0.0) for k in FEATURE_KEYS_A},
                    **{k: set_b.get(k, 0.0) for k in FEATURE_KEYS_B}}

        # Step 3: Run all solvers
        solver_results: Dict[str, Dict[str, Any]] = {}
        cpsat_obj: Optional[float] = None

        for s_name in DATASET_SOLVERS:
            try:
                solver = get_solver(s_name)
                res: SolverResult = solver.solve(
                    problem,
                    compiled_qubo=qubo,
                    time_limit_sec=time_limit_cpsat if s_name == "cpsat" else None,
                )
                solver_results[s_name] = {
                    "status": res.status,
                    "feasible": res.is_feasible,
                    "objective": res.objective_value,
                    "qubo_energy": res.qubo_energy,
                    "time_ms": round(res.wall_time * 1000, 4),
                    "violations": res.num_violations,
                }
                if s_name == "cpsat" and res.is_feasible:
                    cpsat_obj = res.objective_value
            except Exception as e:
                solver_results[s_name] = {
                    "status": "ERROR",
                    "feasible": False,
                    "objective": None,
                    "qubo_energy": None,
                    "time_ms": 0.0,
                    "violations": len(problem.constraints),
                }
                if verbose:
                    print(f"    [WARN] Solver {s_name} failed: {e}")

        # Step 4: Derive labels
        best_time, best_quality, best_combined = _compute_label(solver_results, cpsat_obj)

        row = DatasetRow(
            instance_id=instance_id,
            problem_name=problem.name,
            problem_type=problem_type,
            features=features,
            solver_results=solver_results,
            best_solver_by_time=best_time,
            best_solver_by_quality=best_quality,
            best_solver_combined=best_combined,
            label=best_combined,
        )

        if verbose:
            feasible_count = sum(1 for r in solver_results.values() if r["feasible"])
            print(
                f"  [{instance_id:04d}] {problem_type:12s} | "
                f"vars={qubo.matrix.shape[0]:3d} | "
                f"feasible={feasible_count}/{len(DATASET_SOLVERS)} | "
                f"label={best_combined}"
            )

        return row

    except Exception as e:
        if verbose:
            print(f"  [{instance_id:04d}] ERROR compiling {problem.name}: {e}")
            traceback.print_exc()
        return None


def write_dataset_csv(rows: List[DatasetRow], output_path: str) -> None:
    """Writes all DatasetRow entries to a CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    header = (
        ["instance_id", "problem_name", "problem_type"]
        + ALL_FEATURE_KEYS
        + ALL_SOLVER_COLUMNS
        + ["best_solver_by_time", "best_solver_by_quality", "best_solver_combined", "label"]
    )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            record: Dict[str, Any] = {
                "instance_id": row.instance_id,
                "problem_name": row.problem_name,
                "problem_type": row.problem_type,
                "best_solver_by_time": row.best_solver_by_time,
                "best_solver_by_quality": row.best_solver_by_quality,
                "best_solver_combined": row.best_solver_combined,
                "label": row.label,
            }

            for k in ALL_FEATURE_KEYS:
                record[k] = row.features.get(k, 0.0)

            for s_name in DATASET_SOLVERS:
                r = row.solver_results.get(s_name, {})
                record[f"{s_name}_status"] = r.get("status", "ERROR")
                record[f"{s_name}_feasible"] = int(r.get("feasible", False))
                record[f"{s_name}_objective"] = r.get("objective", "")
                record[f"{s_name}_qubo_energy"] = r.get("qubo_energy", "")
                record[f"{s_name}_time_ms"] = r.get("time_ms", 0.0)
                record[f"{s_name}_violations"] = r.get("violations", 0)

            writer.writerow(record)

    print(f"\n[+] Dataset written to: {output_path}")
    print(f"    Total rows: {len(rows)}")
