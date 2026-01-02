from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier


@dataclass
class OptunaTuneResult:
    best_params: Dict[str, Any]
    best_cv_score: float
    n_trials: int
    seed: int
    study_db: str
    model_path: str
    report_path: str


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def tune_mlp_classifier(
    csv_path: Path,
    target: str,
    out_dir: Path = Path("artifacts/optuna"),
    n_trials: int = 50,
    seed: int = 42,
    cv_folds: int = 5,
    timeout_sec: int | None = None,
) -> OptunaTuneResult:
    """
    Optuna hyperparameter search for an MLPClassifier (classification).
    - Loads RAW CSV (no preprocessing required for Iris; still works generally if features are numeric).
    - Optimizes mean CV accuracy.
    - Persists:
        * SQLite study DB
        * best params JSON
        * best trained model (.pkl)
        * report JSON with best score + params
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in CSV columns: {list(df.columns)}")

    # Basic raw split: features vs target (no scaling here; Iris works fine; MLP may benefit from scaling on other datasets)
    X = df.drop(columns=[target])
    y = df[target]

    # Guard: ensure all features are numeric
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(
            f"Non-numeric feature columns found: {non_numeric}. "
            f"For non-numeric features, add encoding in a pipeline before tuning."
        )

    _ensure_dir(out_dir)
    db_path = out_dir / "study.db"
    storage = f"sqlite:///{db_path.as_posix()}"

    # CV strategy
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        # Layers: 1–3; units per layer: 16–256
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_sizes = tuple(
            trial.suggest_int(f"n_units_l{i}", 16, 256, log=True) for i in range(n_layers)
        )

        # Learning rate + regularization
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 5e-2, log=True)
        alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)

        activation = trial.suggest_categorical("activation", ["relu", "tanh"])
        solver = trial.suggest_categorical("solver", ["adam", "sgd"])

        # Some solvers allow momentum; only meaningful for SGD
        momentum = 0.9
        if solver == "sgd":
            momentum = trial.suggest_float("momentum", 0.0, 0.95)

        clf = MLPClassifier(
            hidden_layer_sizes=hidden_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            momentum=momentum if solver == "sgd" else 0.9,
            max_iter=800,
            early_stopping=True,
            random_state=seed,
        )

        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        return float(np.mean(scores))

    study_name = f"mlp_classifier_{csv_path.stem}_{seed}"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec)

    best_params = dict(study.best_params)
    best_score = float(study.best_value)

    # Train final model on full data using best params
    # Reconstruct hidden sizes from best_params
    n_layers = int(best_params["n_layers"])
    hidden_sizes = tuple(int(best_params[f"n_units_l{i}"]) for i in range(n_layers))

    final_clf = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation=best_params["activation"],
        solver=best_params["solver"],
        alpha=float(best_params["alpha"]),
        learning_rate_init=float(best_params["learning_rate_init"]),
        momentum=float(best_params.get("momentum", 0.9)),
        max_iter=800,
        early_stopping=True,
        random_state=seed,
    )
    final_clf.fit(X, y)

    model_path = out_dir / "best_mlp_model.pkl"
    joblib.dump(final_clf, model_path)

    params_path = out_dir / "best_params.json"
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    report = {
        "best_cv_accuracy": best_score,
        "best_params": best_params,
        "n_trials": int(n_trials),
        "seed": int(seed),
        "cv_folds": int(cv_folds),
        "storage": storage,
        "model_path": str(model_path),
    }
    report_path = out_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return OptunaTuneResult(
        best_params=best_params,
        best_cv_score=best_score,
        n_trials=int(n_trials),
        seed=int(seed),
        study_db=str(db_path),
        model_path=str(model_path),
        report_path=str(report_path),
    )
