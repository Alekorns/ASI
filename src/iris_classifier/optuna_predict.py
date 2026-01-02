from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd


@dataclass
class OptunaPredictResult:
    model_path: str
    input_csv: str
    output_csv: str
    n_rows: int


def predict_with_optuna_model(
    csv_path: Path,
    model_path: Path = Path("artifacts/optuna/best_mlp_model.pkl"),
    target: Optional[str] = None,
    out_csv: Path = Path("artifacts/optuna/predictions.csv"),
) -> OptunaPredictResult:
    """
    Load tuned Optuna model (.pkl) and run predictions on a raw CSV.

    - If target is provided and present in the CSV, it will be dropped from features.
    - Output CSV includes all original columns + 'prediction' column.
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run tune-optuna first to create it."
        )

    df = pd.read_csv(csv_path)

    X = df.copy()
    if target and target in X.columns:
        X = X.drop(columns=[target])

    model = joblib.load(model_path)
    preds = model.predict(X)

    out_df = df.copy()
    out_df["prediction"] = preds

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    return OptunaPredictResult(
        model_path=str(model_path),
        input_csv=str(csv_path),
        output_csv=str(out_csv),
        n_rows=int(len(out_df)),
    )
