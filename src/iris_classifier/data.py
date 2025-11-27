import pandas as pd
from pathlib import Path


def load_iris_df() -> pd.DataFrame:
    base = Path(__file__).resolve().parent.parent.parent
    csv_path = base / "data" / "IRIS.csv"

    df = pd.read_csv(csv_path)

    species_to_id = {name: i for i, name in enumerate(sorted(df["species"].unique()))}
    df["target"] = df["species"].map(species_to_id)

    df = df.drop(columns=["species"])

    return df
