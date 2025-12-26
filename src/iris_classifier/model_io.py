from pathlib import Path
import joblib
from src.iris_classifier.model import IrisClassifier

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def save_model(model: IrisClassifier, name: str) -> None:
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)


def load_model(name: str) -> IrisClassifier:
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model '{name}' not found")
    return joblib.load(path)


def list_models() -> list[str]:
    return [p.stem for p in MODELS_DIR.glob("*.joblib")]
