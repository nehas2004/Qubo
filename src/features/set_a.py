"""
Set A Feature Extractor: Reproduces Volpe et al. (MQT QAO) 9 matrix-level features.
Can run on ANY arbitrary QUBO matrix (NumPy array or dictionary).
"""

from typing import Dict, Union, Tuple
import numpy as np


def extract_set_a_features(
    qubo: Union[np.ndarray, Dict[Tuple[int, int], float]],
    num_vars: int = None,
) -> Dict[str, float]:
    """
    Extracts the 9 matrix features defined in Volpe et al.:
    1. a_num_vars: Variable count (N)
    2. a_num_linear_nonzero: Count of non-zero linear terms (Q_ii != 0)
    3. a_num_quadratic_nonzero: Count of non-zero quadratic terms (Q_ij != 0, i < j)
    4. a_mean_linear: Mean of linear terms
    5. a_var_linear: Variance of linear terms
    6. a_mean_quadratic: Mean of quadratic terms
    7. a_var_quadratic: Variance of quadratic terms
    8. a_mean_all: Mean of all coefficients
    9. a_var_all: Variance of all coefficients
    """
    if isinstance(qubo, dict):
        if num_vars is None:
            max_idx = 0
            for i, j in qubo.keys():
                max_idx = max(max_idx, i, j)
            num_vars = max_idx + 1

        matrix = np.zeros((num_vars, num_vars), dtype=float)
        for (i, j), val in qubo.items():
            matrix[min(i, j), max(i, j)] = val
    else:
        matrix = np.array(qubo, dtype=float)
        num_vars = matrix.shape[0]

    # Linear terms: diagonal Q_ii
    linear_terms = np.diag(matrix)
    nonzero_linear = linear_terms[np.abs(linear_terms) > 1e-12]

    # Quadratic terms: strictly upper-triangular Q_ij (i < j)
    upper_tri_indices = np.triu_indices(num_vars, k=1)
    quadratic_terms = matrix[upper_tri_indices]
    nonzero_quadratic = quadratic_terms[np.abs(quadratic_terms) > 1e-12]

    # All non-zero terms
    all_nonzero = np.concatenate([nonzero_linear, nonzero_quadratic]) if (len(nonzero_linear) + len(nonzero_quadratic)) > 0 else np.array([0.0])

    # Linear stats
    mean_linear = float(np.mean(linear_terms)) if len(linear_terms) > 0 else 0.0
    var_linear = float(np.var(linear_terms)) if len(linear_terms) > 0 else 0.0

    # Quadratic stats
    mean_quad = float(np.mean(quadratic_terms)) if len(quadratic_terms) > 0 else 0.0
    var_quad = float(np.var(quadratic_terms)) if len(quadratic_terms) > 0 else 0.0

    # All coefficients (including zeros in the upper triangle)
    all_upper_coeffs = np.concatenate([linear_terms, quadratic_terms]) if len(linear_terms) > 0 else np.array([0.0])
    mean_all = float(np.mean(all_upper_coeffs))
    var_all = float(np.var(all_upper_coeffs))

    return {
        "a_num_vars": float(num_vars),
        "a_num_linear_nonzero": float(len(nonzero_linear)),
        "a_num_quadratic_nonzero": float(len(nonzero_quadratic)),
        "a_mean_linear": mean_linear,
        "a_var_linear": var_linear,
        "a_mean_quadratic": mean_quad,
        "a_var_quadratic": var_quad,
        "a_mean_all": mean_all,
        "a_var_all": var_all,
    }
