from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import pandas as pd

from iris_classifier.model_io import load_model, save_model

router = APIRouter(tags=["training"])


@router.post("/continue-train")
async def continue_train(
    model_name: str = Form(...),
    new_model_name: str = Form(...),
    train_input: UploadFile = File(...),
):
    try:
        model = load_model(model_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")

    df = pd.read_csv(train_input.file)

    if df.empty:
        raise HTTPException(status_code=400, detail="Training file is empty")

    if "target" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="Training data must contain 'target'",
        )

    X = df.drop(columns=["target"])
    y = df["target"]

    metrics = model.train(X, y)  # should return dict of metrics
    save_model(model, new_model_name)

    return {
        "new_model_name": new_model_name,
        "metrics": metrics,
    }
