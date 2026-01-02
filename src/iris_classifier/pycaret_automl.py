from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


@dataclass(frozen=True)
class PyCaretArtifacts:
    best_model_name: str
    metrics: Dict[str, float]
    model_path: Path
    predictions_path: Path


def train_pycaret_classifier(
    csv_path: Path,
    target: str,
    out_dir: Path = Path("artifacts/pycaret"),
    test_size: float = 0.2,
    seed: int = 42,
) -> PyCaretArtifacts:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")

    out_dir = Path(out_dir)
    model_dir = out_dir / "models"
    pred_dir = out_dir / "predictions"
    report_dir = out_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Holdout split (raw)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[target],
    )

    from pycaret.classification import setup, compare_models, finalize_model, predict_model, save_model, pull

    setup(
        data=train_df,
        target=target,
        session_id=seed,
        verbose=False,
    )

    best = compare_models()
    compare_table = pull()
    compare_table.to_csv(report_dir / "compare_models.csv", index=False)

    final_best = finalize_model(best)

    preds = predict_model(final_best, data=test_df)

    y_true = test_df[target]
    y_pred = preds["prediction_label"]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }

    predictions_path = pred_dir / "holdout_predictions.csv"
    preds.to_csv(predictions_path, index=False)

    model_basename = model_dir / "best_model"
    save_model(final_best, str(model_basename))
    model_path = Path(str(model_basename) + ".pkl")

    return PyCaretArtifacts(
        best_model_name=type(best).__name__,
        metrics=metrics,
        model_path=model_path,
        predictions_path=predictions_path,
    )


def predict_with_pycaret_model(
    model_path: Path,
    csv_path: Path,
    out_path: Path = Path("artifacts/pycaret/predictions/inference_predictions.csv"),
) -> Path:

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    from pycaret.classification import load_model, predict_model


    basename = str(model_path).removesuffix(".pkl")
    model = load_model(basename)

    preds = predict_model(model, data=df)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False)
    return out_path
