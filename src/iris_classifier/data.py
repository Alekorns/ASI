from sklearn.datasets import load_iris
import pandas as pd


def load_iris_df() -> pd.DataFrame:

    iris = load_iris(as_frame=True)
    return iris.frame.copy()