from typing import Iterable

import numpy as np
import pandas as pd
from sklearn import preprocessing

from util import HyperParam


class Transformer:
    def __init__(self, df: pd.DataFrame, discrete_columns: Iterable[str]):
        assert set(discrete_columns).issubset(df.columns)
        dfc = df.copy()
        self._dcols = list(discrete_columns)
        self._all_cols = df.columns.copy()
        self._ord_enc = preprocessing.OrdinalEncoder()
        dfc[self._dcols] = self._ord_enc.fit_transform(dfc[self._dcols])
        self._scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        self._scalar.fit(dfc)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        dfc = df.copy()
        dfc[self._dcols] = self._ord_enc.transform(dfc[self._dcols])
        return self._scalar.transform(dfc)

    def inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        data = self._scalar.inverse_transform(data)
        # todo: only round discrete columns & integer continuous columns
        df = pd.DataFrame(np.rint(data).astype(np.int64), columns=self._all_cols)
        # todo: IndexError may raise
        df[self._dcols] = self._ord_enc.inverse_transform(df[self._dcols])
        return df

    def low_level(self):
        return self._ord_enc, self._scalar


def fit_transform(opt: HyperParam, data_path: str, discrete_columns: Iterable[str]):
    df = pd.read_csv(data_path)
    transformer = Transformer(df, discrete_columns)
    raw_dataset = transformer.transform(df)
    opt.dataset_size = raw_dataset.shape[0]
    opt.feature_dim = raw_dataset.shape[1]
    return transformer, raw_dataset
