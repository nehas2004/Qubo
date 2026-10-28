"""
Benchmark package: problem generation and dataset collection for ML training.
"""
from .generator import (
    generate_job_shop,
    generate_knapsack,
    generate_max_cut,
    generate_set_cover,
    generate_graph_coloring,
    build_instance_schedule,
    build_problem_from_spec,
    InstanceSpec,
)
from .runner import (
    run_instance,
    write_dataset_csv,
    DatasetRow,
    DATASET_SOLVERS,
    ALL_FEATURE_KEYS,
)

__all__ = [
    "generate_job_shop",
    "generate_knapsack",
    "generate_max_cut",
    "generate_set_cover",
    "generate_graph_coloring",
    "build_instance_schedule",
    "build_problem_from_spec",
    "InstanceSpec",
    "run_instance",
    "write_dataset_csv",
    "DatasetRow",
    "DATASET_SOLVERS",
    "ALL_FEATURE_KEYS",
]
