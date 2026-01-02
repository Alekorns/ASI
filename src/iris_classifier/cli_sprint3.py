from __future__ import annotations

from pathlib import Path
from typing import Optional
from iris_classifier.optuna_tuning import tune_mlp_classifier
from iris_classifier.optuna_predict import predict_with_optuna_model



import typer

from iris_classifier.pycaret_automl import train_pycaret_classifier, predict_with_pycaret_model

app = typer.Typer(help="ASI Sprint 3 - PyCaret CLI")


@app.command("train-pycaret")
def train_pycaret(
    csv: Path = typer.Option(..., "--csv", help="Raw training CSV path"),
    target: str = typer.Option(..., "--target", help="Target column name"),
    out_dir: Path = typer.Option(Path("artifacts/pycaret"), "--out", help="Output directory"),
    test_size: float = typer.Option(0.2, "--test-size", help="Holdout fraction"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
):
    res = train_pycaret_classifier(
        csv_path=csv,
        target=target,
        out_dir=out_dir,
        test_size=test_size,
        seed=seed,
    )
    typer.echo("PyCaret training completed.")
    typer.echo(f"Best model: {res.best_model_name}")
    typer.echo("Metrics:")
    for k, v in res.metrics.items():
        typer.echo(f"  {k}: {v:.6f}")
    typer.echo(f"Model saved: {res.model_path}")
    typer.echo(f"Predictions saved: {res.predictions_path}")


@app.command("tune-optuna")
def tune_optuna(
    csv: Path = typer.Option(..., "--csv", help="Path to raw CSV"),
    target: str = typer.Option(..., "--target", help="Target column name"),
    out_dir: Path = typer.Option(Path("artifacts/optuna"), "--out", help="Output directory"),
    n_trials: int = typer.Option(50, "--n-trials", help="Number of Optuna trials"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    cv_folds: int = typer.Option(5, "--cv-folds", help="Stratified K-folds"),
    timeout_sec: int = typer.Option(0, "--timeout-sec", help="Timeout seconds (0 = no timeout)"),
):
    """
    Optuna: tune MLP hyperparameters (layers/neurons/learning rate) on raw data.
    """
    res = tune_mlp_classifier(
        csv_path=csv,
        target=target,
        out_dir=out_dir,
        n_trials=n_trials,
        seed=seed,
        cv_folds=cv_folds,
        timeout_sec=None if timeout_sec <= 0 else timeout_sec,
    )

    typer.echo("Optuna tuning completed.")
    typer.echo(f"Best CV accuracy: {res.best_cv_score:.6f}")
    typer.echo(f"Best params: {res.best_params}")
    typer.echo(f"Study DB: {res.study_db}")
    typer.echo(f"Model saved: {res.model_path}")
    typer.echo(f"Report saved: {res.report_path}")

@app.command("predict-optuna")
def predict_optuna(
    csv: Path = typer.Option(..., "--csv", help="CSV to run predictions on"),
    model: Path = typer.Option(Path("artifacts/optuna/best_mlp_model.pkl"), "--model", help="Path to tuned Optuna model"),
    target: str = typer.Option("", "--target", help="Optional: target column to drop (e.g. species). Leave empty if unknown."),
    out_csv: Path = typer.Option(Path("artifacts/optuna/predictions.csv"), "--out", help="Where to save predictions CSV"),
):

    res = predict_with_optuna_model(
        csv_path=csv,
        model_path=model,
        target=target if target.strip() else None,
        out_csv=out_csv,
    )

    typer.echo("Optuna prediction completed.")
    typer.echo(f"Input: {res.input_csv}")
    typer.echo(f"Model: {res.model_path}")
    typer.echo(f"Rows: {res.n_rows}")
    typer.echo(f"Predictions saved: {res.output_csv}")



@app.command("predict-pycaret")
def predict_pycaret(
    model: Path = typer.Option(..., "--model", help="Path to saved PyCaret model .pkl"),
    csv: Path = typer.Option(..., "--csv", help="Raw inference CSV path"),
    out: Path = typer.Option(Path("artifacts/pycaret/predictions/inference_predictions.csv"), "--out", help="Output CSV"),
):
    out_path = predict_with_pycaret_model(model_path=model, csv_path=csv, out_path=out)
    typer.echo(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    app()
