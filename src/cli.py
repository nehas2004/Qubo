"""
Command-Line Interface (CLI) for Qubo Optimizer & DSL Compiler.
"""

import argparse
import sys
import json
import numpy as np

from src.dsl.parser import DSLParser
from src.dsl.compiler import QUBOCompiler
from src.dsl.verifier import QUBOVerifier
from src.features.set_a import extract_set_a_features
from src.solvers import (
    get_solver,
    list_solvers,
    benchmark_solvers,
    SolverResult,
)
from src.models.selector import SolverSelector


def main():
    parser = argparse.ArgumentParser(
        prog="qubo",
        description="DSL Compiler, Verifier, and Feature Extractor for QUBO Solver Selection",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: compile
    compile_p = subparsers.add_parser("compile", help="Compile a .opt file to QUBO and extract features")
    compile_p.add_argument("file", help="Path to .opt problem file")
    compile_p.add_argument("--penalty", type=float, default=None, help="Explicit penalty multiplier")
    compile_p.add_argument("--json", action="store_true", help="Output all features as JSON")

    # Command: verify
    verify_p = subparsers.add_parser("verify", help="Verify QUBO compilation against CP-SAT ground truth")
    verify_p.add_argument("file", help="Path to .opt problem file")
    verify_p.add_argument("--penalty", type=float, default=None, help="Explicit penalty multiplier")
    verify_p.add_argument("--max-vars", type=int, default=20, help="Max variables for exhaustive scan (default: 20)")
    verify_p.add_argument("--json", action="store_true", help="Output verification report as JSON")

    # Command: list-solvers
    subparsers.add_parser("list-solvers", help="List all available Classical, Quantum-Inspired, and Quantum Solvers")

    # Command: solve
    solve_p = subparsers.add_parser("solve", help="Solve a .opt problem using a chosen solver")
    solve_p.add_argument("file", help="Path to .opt problem file")
    solve_p.add_argument(
        "--solver",
        type=str,
        default="cpsat",
        help="Solver to use: cpsat, sa, tabu, sqa, sbm, qpu, auto (default: cpsat). "
             "Use 'auto' to let the ML model predict the best solver.",
    )
    solve_p.add_argument("--penalty", type=float, default=None, help="Explicit penalty multiplier")
    solve_p.add_argument("--reads", type=int, default=None, help="Number of reads / shots / samples")
    solve_p.add_argument("--json", action="store_true", help="Output result as JSON")

    # Command: benchmark
    bench_p = subparsers.add_parser("benchmark", help="Benchmark multiple solvers side-by-side on a .opt problem")
    bench_p.add_argument("file", help="Path to .opt problem file")
    bench_p.add_argument(
        "--solvers",
        type=str,
        default="cpsat,sa,tabu,sqa,sbm",
        help="Comma-separated solver list (default: cpsat,sa,tabu,sqa,sbm)",
    )
    bench_p.add_argument("--penalty", type=float, default=None, help="Explicit penalty multiplier")
    bench_p.add_argument("--json", action="store_true", help="Output benchmark comparison as JSON")

    args = parser.parse_args()

    if args.command == "list-solvers":
        print("\n" + "=" * 65)
        print(f"{'SOLVER':12s} | {'CATEGORY':20s} | {'DESCRIPTION'}")
        print("=" * 65)
        for name, info in list_solvers().items():
            print(f"{name:12s} | {info['category']:20s} | {info['description']}")
        print("=" * 65)

    elif args.command == "compile":
        dsl_parser = DSLParser()
        print(f"[*] Reading and parsing '{args.file}'...")
        try:
            problem = dsl_parser.parse_file(args.file)
        except Exception as e:
            print(f"[ERROR] Failed to parse: {e}")
            sys.exit(1)

        print(f"[+] Problem '{problem.name}' parsed successfully.")
        print(f"    - Declared variable groups: {list(problem.variables.keys())}")
        print(f"    - Constraints count: {len(problem.constraints)}")

        # Compile to QUBO
        compiler = QUBOCompiler(default_penalty=args.penalty)
        compiled = compiler.compile(problem)

        print(f"\n[+] Lowered to QUBO Matrix:")
        print(f"    - Original variables: {compiled.num_orig_vars}")
        print(f"    - Slack variables added: {compiled.num_slack_vars}")
        print(f"    - Total matrix dimension: {compiled.matrix.shape[0]} x {compiled.matrix.shape[1]}")
        print(f"    - Penalty multiplier P: {compiled.penalty_multiplier:.2f}")
        print(f"    - Energy constant offset: {compiled.constant_offset:.2f}")

        # Extract Set A
        set_a = extract_set_a_features(compiled.matrix)
        set_b = compiled.set_b_features

        if args.json:
            combined = {"set_a": set_a, "set_b": set_b}
            print(json.dumps(combined, indent=2))
        else:
            print("\n" + "=" * 50)
            print("SET A FEATURES (Volpe et al. 9 Matrix Stats)")
            print("=" * 50)
            for k, v in set_a.items():
                print(f"  {k:30s} : {v:.4f}")

            print("\n" + "=" * 50)
            print("SET B FEATURES (Our Language & Slack Additions)")
            print("=" * 50)
            for k, v in set_b.items():
                print(f"  {k:30s} : {v:.4f}")

    elif args.command == "verify":
        dsl_parser = DSLParser()
        print(f"[*] Reading and parsing '{args.file}'...")
        try:
            problem = dsl_parser.parse_file(args.file)
        except Exception as e:
            print(f"[ERROR] Failed to parse: {e}")
            sys.exit(1)

        print(f"[+] Parsed '{problem.name}'. Running CP-SAT ground truth and QUBO verification...")
        verifier = QUBOVerifier(max_brute_force_vars=args.max_vars)
        report = verifier.verify(problem, penalty=args.penalty)

        if args.json:
            rep_dict = {
                "problem_name": report.problem_name,
                "is_valid": report.is_valid,
                "num_orig_vars": report.num_orig_vars,
                "num_slack_vars": report.num_slack_vars,
                "total_qubo_vars": report.total_qubo_vars,
                "penalty_multiplier": report.penalty_multiplier,
                "native_status": report.native_status,
                "native_opt_obj": report.native_opt_obj,
                "qubo_ground_energy": report.qubo_ground_energy,
                "spectral_gap": report.spectral_gap,
                "ground_state_feasible": report.ground_state_feasible,
                "num_ground_states": report.num_ground_states,
                "all_feasible_count": report.all_feasible_count,
                "all_infeasible_count": report.all_infeasible_count,
                "exhaustively_verified": report.exhaustively_verified,
                "wall_time_sec": report.wall_time_sec,
                "messages": report.messages,
            }
            print(json.dumps(rep_dict, indent=2))
        else:
            print("\n" + report.summary())

        if not report.is_valid:
            sys.exit(1)

    elif args.command == "solve":
        dsl_parser = DSLParser()
        try:
            problem = dsl_parser.parse_file(args.file)
        except Exception as e:
            print(f"[ERROR] Failed to parse '{args.file}': {e}")
            sys.exit(1)

        # Compile QUBO once here (needed for auto-selection and solve)
        compiler = QUBOCompiler(default_penalty=args.penalty)
        compiled_qubo = compiler.compile(problem)

        # --- AUTO SOLVER SELECTION ---
        chosen_solver_name = args.solver
        auto_proba: dict = {}
        if args.solver.lower() == "auto":
            try:
                selector = SolverSelector()
                chosen_solver_name = selector.predict(compiled_qubo)
                auto_proba = selector.predict_proba(compiled_qubo)
                if not args.json:
                    print(f"[ML] Auto-selected solver: {chosen_solver_name}")
                    print("     Confidence scores:")
                    for s_name, prob in auto_proba.items():
                        bar = "|" * int(prob * 30)
                        print(f"       {s_name:8s}: {prob:.2%}  {bar}")
                    print()
            except FileNotFoundError as e:
                print(f"[ERROR] --solver auto requires a trained model.")
                print(f"  {e}")
                sys.exit(1)

        try:
            solver = get_solver(chosen_solver_name)
        except ValueError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

        solver_kwargs = {}
        if args.reads is not None:
            solver_kwargs["num_reads"] = args.reads
            solver_kwargs["num_shots"] = args.reads

        result = solver.solve(problem, compiled_qubo=compiled_qubo, penalty=args.penalty, **solver_kwargs)

        if args.json:
            out = {
                "solver_name": result.solver_name,
                "auto_selected": args.solver.lower() == "auto",
                "auto_confidence": auto_proba,
                "status": result.status,
                "is_feasible": result.is_feasible,
                "objective_value": result.objective_value,
                "qubo_energy": result.qubo_energy,
                "wall_time_sec": result.wall_time,
                "assignment": result.assignment,
                "num_violations": result.num_violations,
                "max_violation": result.max_violation,
                "solver_info": result.solver_info,
            }
            print(json.dumps(out, indent=2))
        else:
            print("\n" + "=" * 55)
            print(f"SOLVER EXECUTION: {result.solver_name}")
            print("=" * 55)
            print(f"Status           : {result.status}")
            print(f"Feasibility      : {'FEASIBLE' if result.is_feasible else f'INFEASIBLE ({result.num_violations} violations)'}")
            print(f"Objective Value  : {result.objective_value if result.objective_value is not None else 'N/A'}")
            if result.qubo_energy is not None:
                print(f"QUBO Energy      : {result.qubo_energy:.4f}")
            print(f"Execution Time   : {result.wall_time * 1000:.2f} ms")
            print("\nVariable Assignment:")
            for k in sorted(result.assignment.keys()):
                print(f"  {k:15s} = {result.assignment[k]}")

    elif args.command == "benchmark":
        dsl_parser = DSLParser()
        try:
            problem = dsl_parser.parse_file(args.file)
        except Exception as e:
            print(f"[ERROR] Failed to parse '{args.file}': {e}")
            sys.exit(1)

        solvers_list = [s.strip() for s in args.solvers.split(",") if s.strip()]
        results = benchmark_solvers(problem, solvers=solvers_list, penalty=args.penalty)

        if args.json:
            out = [
                {
                    "solver": r.solver_name,
                    "status": r.status,
                    "is_feasible": r.is_feasible,
                    "objective": r.objective_value,
                    "qubo_energy": r.qubo_energy,
                    "wall_time_sec": r.wall_time,
                    "assignment": r.assignment,
                    "violations": r.num_violations,
                }
                for r in results
            ]
            print(json.dumps(out, indent=2))
        else:
            print("\n" + "=" * 75)
            print(f"{'SOLVER':28s} | {'STATUS':8s} | {'FEASIBLE':8s} | {'OBJECTIVE':9s} | {'TIME (ms)':9s}")
            print("=" * 75)
            for r in results:
                obj_str = f"{r.objective_value:.4f}" if r.objective_value is not None else "N/A"
                feas_str = "YES" if r.is_feasible else f"NO ({r.num_violations})"
                print(f"{r.solver_name:28s} | {r.status:8s} | {feas_str:8s} | {obj_str:9s} | {r.wall_time * 1000:8.2f}ms")
            print("=" * 75)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
