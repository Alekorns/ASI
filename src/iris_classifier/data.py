import pandas as pd
from pathlib import Path


def load_iris_df(path: str = "data/IRIS.csv") -> pd.DataFrame:

    path = Path(path)
    df = pd.read_csv(path)

    species_to_id = {name: i for i, name in enumerate(sorted(df["species"].unique()))}
    df["target"] = df["species"].map(species_to_id)

    return df
