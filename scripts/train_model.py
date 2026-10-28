"""
ML Solver Selection Model Trainer.

Trains an XGBoost multi-class classifier to predict the best QUBO solver
(cpsat, sa, tabu, sqa, sbm) from the compiled QUBO's feature vector.

Usage:
    python -m scripts.train_model [--data PATH] [--model-out PATH] [--report]

Features used:
    - Set A (9 features): Matrix statistics (Volpe et al. 9 features)
    - Set B (15 features): Graph density, slack ratio, constraint structure

Label: best_solver_combined (solver with lowest time × (1 + optimality gap))
"""

import argparse
import os
import sys
import json
import pickle
import csv
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark.runner import ALL_FEATURE_KEYS, DATASET_SOLVERS


def load_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Loads dataset.csv and returns X (feature matrix), y (integer labels),
    label_names (solver names), and feature_names.
    """
    rows = []
    labels_raw = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("label", "none") in ("none", "", "ERROR"):
                continue
            feat_vec = []
            for k in ALL_FEATURE_KEYS:
                try:
                    feat_vec.append(float(row.get(k, 0.0) or 0.0))
                except (ValueError, TypeError):
                    feat_vec.append(0.0)
            rows.append(feat_vec)
            labels_raw.append(row["label"])

    if not rows:
        raise ValueError(f"No valid rows found in {csv_path}.")

    # Encode labels as integers
    unique_labels = sorted(set(labels_raw))
    label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}

    X = np.array(rows, dtype=float)
    y = np.array([label_to_int[lbl] for lbl in labels_raw], dtype=int)

    print(f"  Loaded {len(X)} rows, {X.shape[1]} features, {len(unique_labels)} classes.")
    print(f"  Label mapping: {label_to_int}")

    return X, y, unique_labels, ALL_FEATURE_KEYS


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    label_names: List[str],
    feature_names: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains an XGBoost classifier with cross-validation and returns a model bundle.
    """
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )
    from sklearn.preprocessing import StandardScaler

    print("\n[*] Splitting dataset (80/20 train-test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

    # XGBoost handles raw features well; optional scaling for reproducibility
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_classes = len(label_names)
    print(f"\n[*] Training XGBClassifier ({n_classes} classes)...")

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False,
    )

    # Evaluation
    y_pred_test = clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred_test)

    # 5-Fold stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring="accuracy")

    print(f"\n[+] Training complete.")
    print(f"    Test Accuracy     : {test_acc * 100:.2f}%")
    print(f"    CV Accuracy (5-fold): {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

    print("\n  Classification Report:")
    class_names = [label_names[i] for i in sorted(set(y_test))]
    print(classification_report(y_test, y_pred_test, target_names=class_names))

    # Feature importances
    importances = clf.feature_importances_
    top_features = sorted(
        zip(feature_names, importances.tolist()),
        key=lambda x: -x[1],
    )[:10]
    print("  Top 10 Feature Importances:")
    for feat, imp in top_features:
        bar = "|" * int(imp * 100)
        print(f"    {feat:35s}: {imp:.4f} {bar}")

    return {
        "classifier": clf,
        "scaler": scaler,
        "label_names": label_names,
        "feature_names": feature_names,
        "label_to_int": {lbl: i for i, lbl in enumerate(label_names)},
        "int_to_label": {i: lbl for i, lbl in enumerate(label_names)},
        "test_accuracy": float(test_acc),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "feature_importances": {feat: float(imp) for feat, imp in zip(feature_names, importances)},
    }


def save_model(bundle: Dict[str, Any], model_path: str) -> None:
    """Saves the model bundle as a pickle file."""
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)

    # Split into pickle (clf + scaler) and JSON (metadata)
    pkl_path = model_path if model_path.endswith(".pkl") else model_path + ".pkl"
    meta_path = pkl_path.replace(".pkl", "_metadata.json")

    with open(pkl_path, "wb") as f:
        pickle.dump({
            "classifier": bundle["classifier"],
            "scaler": bundle["scaler"],
        }, f)

    metadata = {
        "label_names": bundle["label_names"],
        "feature_names": bundle["feature_names"],
        "label_to_int": bundle["label_to_int"],
        "int_to_label": {str(k): v for k, v in bundle["int_to_label"].items()},
        "test_accuracy": bundle["test_accuracy"],
        "cv_accuracy_mean": bundle["cv_accuracy_mean"],
        "cv_accuracy_std": bundle["cv_accuracy_std"],
        "feature_importances": bundle["feature_importances"],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[+] Model saved to: {os.path.abspath(pkl_path)}")
    print(f"[+] Metadata saved to: {os.path.abspath(meta_path)}")


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost QUBO solver selection model.")
    parser.add_argument(
        "--data", type=str, default="data/dataset.csv",
        help="Path to the dataset CSV (default: data/dataset.csv).",
    )
    parser.add_argument(
        "--model-out", type=str, default="models/solver_selector.pkl",
        help="Output path for the trained model pickle (default: models/solver_selector.pkl).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of dataset reserved for testing (default: 0.2).",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  QUBO Solver Selector: XGBoost Classifier Trainer")
    print("=" * 65)
    print(f"  Dataset : {args.data}")
    print(f"  Output  : {args.model_out}")
    print("=" * 65)

    if not os.path.exists(args.data):
        print(f"\n[ERROR] Dataset not found: {args.data}")
        print("  Run first: python -m scripts.generate_dataset")
        sys.exit(1)

    print("\n[*] Loading dataset...")
    X, y, label_names, feature_names = load_dataset(args.data)

    if len(X) < 20:
        print(f"[WARN] Only {len(X)} rows found. Need ≥ 20 for meaningful training.")
        print("  Run: python -m scripts.generate_dataset --instances 80")
        if len(X) < 5:
            sys.exit(1)

    bundle = train_model(X, y, label_names, feature_names, test_size=args.test_size)
    save_model(bundle, args.model_out)

    print("\n[->] Next step: python -m src.cli solve problem.opt --solver auto")


if __name__ == "__main__":
    main()
