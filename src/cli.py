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

    args = parser.parse_args()

    if args.command == "compile":
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

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
