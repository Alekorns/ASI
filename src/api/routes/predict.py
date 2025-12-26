from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import pandas as pd

from iris_classifier.model_io import load_model

router = APIRouter(tags=["prediction"])


@router.post("/predict")
async def predict(
    model_name: str = Form(...),
    input: UploadFile = File(...),
):
    try:
        model = load_model(model_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")

    df = pd.read_csv(input.file)

    if "target" in df.columns:
        raise HTTPException(
            status_code=400,
            detail="Input must NOT contain 'target'",
        )

    preds = model.predict(df.to_numpy())

    return {
        "model_name": model_name,
        "predictions": preds.tolist(),
    }
