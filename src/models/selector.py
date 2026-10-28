"""
Solver Selector: ML inference module for automatic solver recommendation.

Loads the trained XGBoost model and predicts the best solver for a given
compiled QUBO problem based on its feature vector (Set A + Set B).

Usage:
    from src.models.selector import SolverSelector

    selector = SolverSelector()          # loads default model
    solver_name = selector.predict(qubo) # returns e.g. "sa"
    proba = selector.predict_proba(qubo) # returns {"cpsat": 0.9, "sa": 0.08, ...}
"""

import os
import json
import pickle
from typing import Optional, Dict, List

import numpy as np

from src.dsl.compiler import CompiledQUBO
from src.features.set_a import extract_set_a_features
from src.benchmark.runner import ALL_FEATURE_KEYS

# Default path relative to project root
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "solver_selector.pkl",
)
_DEFAULT_META_PATH = _DEFAULT_MODEL_PATH.replace(".pkl", "_metadata.json")


class SolverSelector:
    """
    Loads the trained XGBoost solver-selection model and provides
    predict() / predict_proba() for a compiled QUBO.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Path to the .pkl model file. Defaults to
                        models/solver_selector.pkl relative to project root.
        """
        self.model_path = model_path or _DEFAULT_MODEL_PATH
        self.meta_path = self.model_path.replace(".pkl", "_metadata.json")

        self._clf = None
        self._scaler = None
        self._label_names: List[str] = []
        self._int_to_label: Dict[int, str] = {}
        self._feature_names: List[str] = []
        self._loaded = False

        self._load()

    def _load(self) -> None:
        """Loads the model pickle and metadata JSON."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"No trained model found at: {self.model_path}\n"
                f"Run first: python -m scripts.train_model"
            )

        with open(self.model_path, "rb") as f:
            bundle = pickle.load(f)

        self._clf = bundle["classifier"]
        self._scaler = bundle["scaler"]

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._label_names = meta["label_names"]
            self._int_to_label = {int(k): v for k, v in meta["int_to_label"].items()}
            self._feature_names = meta["feature_names"]
        else:
            # Fallback: derive from classifier
            self._label_names = [str(i) for i in range(self._clf.n_classes_)]
            self._int_to_label = {i: str(i) for i in range(self._clf.n_classes_)}
            self._feature_names = ALL_FEATURE_KEYS

        self._loaded = True

    def _extract_features(self, qubo: CompiledQUBO) -> np.ndarray:
        """Builds the 24-feature vector from a compiled QUBO."""
        set_a = extract_set_a_features(qubo.matrix)
        set_b = qubo.set_b_features

        feat_dict = {**set_a, **set_b}
        vec = []
        for k in self._feature_names:
            try:
                vec.append(float(feat_dict.get(k, 0.0) or 0.0))
            except (TypeError, ValueError):
                vec.append(0.0)

        return np.array(vec, dtype=float).reshape(1, -1)

    def predict(self, qubo: CompiledQUBO) -> str:
        """
        Predicts the best solver name for the given compiled QUBO.

        Args:
            qubo: A CompiledQUBO instance from QUBOCompiler.compile().

        Returns:
            Solver name string, e.g. "cpsat", "sa", "sbm".
        """
        X = self._extract_features(qubo)
        X_scaled = self._scaler.transform(X)
        pred_int = int(self._clf.predict(X_scaled)[0])
        return self._int_to_label.get(pred_int, "cpsat")

    def predict_proba(self, qubo: CompiledQUBO) -> Dict[str, float]:
        """
        Returns prediction probabilities for all solver classes.

        Args:
            qubo: A CompiledQUBO instance from QUBOCompiler.compile().

        Returns:
            Dict mapping solver name -> probability (0.0–1.0), sorted desc.
        """
        X = self._extract_features(qubo)
        X_scaled = self._scaler.transform(X)
        proba = self._clf.predict_proba(X_scaled)[0]

        result = {
            self._int_to_label.get(i, str(i)): float(p)
            for i, p in enumerate(proba)
        }
        return dict(sorted(result.items(), key=lambda x: -x[1]))

    @property
    def label_names(self) -> List[str]:
        return self._label_names

    @property
    def model_accuracy(self) -> Optional[float]:
        """Returns test accuracy from saved metadata, if available."""
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta.get("test_accuracy")
        return None
