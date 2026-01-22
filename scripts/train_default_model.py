from src.iris_classifier.data import load_iris_df
from src.iris_classifier.model import IrisClassifier
from src.iris_classifier.model_io import save_model


def main() -> None:
    df = load_iris_df()
    model = IrisClassifier()
    model.train(df)
    save_model(model, "default")
    print("Saved models/default.joblib")


if __name__ == "__main__":
    main()
