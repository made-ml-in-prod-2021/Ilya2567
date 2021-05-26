from sklearn.base import BaseEstimator, TransformerMixin
from itertools import count
import pandas as pd
import numpy as np


class BinaryEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, ignore_unknown: bool = False, dtype: str = 'float'):
        super().__init__()
        self._ignore_unknown = ignore_unknown
        self._dtype = dtype
        self._encoder = []

    def get_params(self, **kwarg):
        return {'ignore_unknown': self._ignore_unknown, 'dtype': self._dtype}

    def fit(self, x, y=None):
        if isinstance(x, (pd.DataFrame, pd.Series)):
            x = x.to_numpy()

        for i in range(x.shape[1]):
            unique = np.unique(x[:, i])
            if len(unique) != 2:
                raise ValueError(f'Feature at position {i} is not binary')
            enc = {val: i for i, val in enumerate(unique)}
            self._encoder.append(enc)

    def transform(self, features):
        if features.shape[1] != len(self._encoder):
            raise ValueError(f'Expected {len(self._encoder)} features, got {features.shape[1]}.')

        if isinstance(features, (pd.DataFrame, pd.Series)):
            features = features.to_numpy()

        bin_features = np.zeros(features.shape, dtype=self._dtype)
        for row, features_row in zip(count(), features):
            for col, value, enc in zip(count(), features_row, self._encoder):
                if not self._ignore_unknown and value not in enc:
                    raise ValueError(f'Found new unique values {value} which does not exist in train')

                bin_features[row, col] = enc[value]

        return bin_features

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x, y)
        return self.transform(x)
