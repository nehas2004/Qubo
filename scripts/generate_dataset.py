"""
Dataset Generation Script.

Generates a labeled QUBO solver selection dataset by:
  1. Creating parametric optimization problem instances across 5 problem types.
  2. Running all solvers (CP-SAT, SA, Tabu, SQA, SBM) on each instance.
  3. Extracting Set A + Set B features.
  4. Writing the results to data/dataset.csv.

Usage:
    python -m scripts.generate_dataset [--instances N] [--output PATH] [--verbose]

Options:
    --instances N       Number of instances per problem type (default: 80, total ~400)
    --output PATH       Output CSV path (default: data/dataset.csv)
    --verbose           Print per-instance progress
    --smoke-test        Quick run with 5 instances per type for validation
"""

import argparse
import sys
import os
import time

# Ensure project root is in path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark.generator import build_instance_schedule, build_problem_from_spec
from src.benchmark.runner import run_instance, write_dataset_csv, DatasetRow


def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate the QUBO solver selection dataset."
    )
    parser.add_argument(
        "--instances", type=int, default=80,
        help="Number of instances per problem type (default: 80). Total = N × 5 types.",
    )
    parser.add_argument(
        "--output", type=str, default="data/dataset.csv",
        help="Path for the output CSV file (default: data/dataset.csv).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-instance progress lines.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick validation run: 5 instances per type (25 total).",
    )
    args = parser.parse_args()

    n_per_type = 5 if args.smoke_test else args.instances
    output_path = args.output

    print("=" * 65)
    print("  QUBO Solver Selection Dataset Generator")
    print("=" * 65)
    print(f"  Instances per problem type : {n_per_type}")
    print(f"  Problem types              : 5 (JobShop, Knapsack, MaxCut, SetCover, GraphColor)")
    print(f"  Total instances planned    : {n_per_type * 5}")
    print(f"  Solvers                    : cpsat, sa, tabu, sqa, sbm")
    print(f"  Output CSV                 : {output_path}")
    print("=" * 65)

    # 1. Build the instance schedule
    specs = build_instance_schedule(n_instances_per_type=n_per_type, rng_seed=42)
    total = len(specs)
    print(f"\n[*] Generating {total} problem instances...\n")

    # 2. Run each instance
    rows: list[DatasetRow] = []
    start_time = time.perf_counter()
    failed = 0
    skipped = 0

    for idx, spec in enumerate(specs):
        if args.verbose:
            print(f"  [{idx+1:04d}/{total}] type={spec.problem_type:12s} params={spec.params}")

        try:
            problem = build_problem_from_spec(spec)
        except Exception as e:
            print(f"  [WARN] Could not generate instance {idx}: {e}")
            skipped += 1
            continue

        row = run_instance(
            instance_id=idx,
            problem=problem,
            problem_type=spec.problem_type,
            verbose=args.verbose,
        )

        if row is None:
            failed += 1
            continue

        rows.append(row)

        # Progress report every 25 instances
        if (idx + 1) % 25 == 0:
            elapsed = time.perf_counter() - start_time
            rate = (idx + 1) / elapsed
            eta = (total - idx - 1) / rate if rate > 0 else 0
            print(
                f"  Progress: {idx+1}/{total} ({100*(idx+1)/total:.0f}%) | "
                f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s | "
                f"Collected: {len(rows)} rows"
            )

    total_elapsed = time.perf_counter() - start_time

    print(f"\n[+] Run complete in {total_elapsed:.1f}s")
    print(f"    Successful rows : {len(rows)}")
    print(f"    Failed          : {failed}")
    print(f"    Skipped (gen)   : {skipped}")

    if not rows:
        print("[ERROR] No rows collected. Check solver installation.")
        sys.exit(1)

    # 3. Label distribution summary
    label_counts: dict[str, int] = {}
    for r in rows:
        label_counts[r.label] = label_counts.get(r.label, 0) + 1
    valid_count = sum(c for lbl, c in label_counts.items() if lbl not in ("none", ""))
    print("\n  Label Distribution (best_solver_combined):")
    for solver, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(rows)
        bar = "|" * int(pct / 2)
        print(f"    {solver:28s}: {count:4d} ({pct:5.1f}%) {bar}")
    print(f"\n  Usable rows (feasible label): {valid_count} / {len(rows)}")

    # 4. Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    write_dataset_csv(rows, output_path)

    print(f"\n[OK] Dataset ready at: {os.path.abspath(output_path)}")
    print("[->] Next step: run  python -m scripts.train_model  to train the XGBoost selector.")


if __name__ == "__main__":
    main()
