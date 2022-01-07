import pandas as pd
import numpy as np


def normalize(X, _max, _min):
    return (X - _min) / (_max - _min)


def reverse_normalize(X, _max, _min):
    return X * (_max - _min) + _min


def preprocess_timeseries(
    _timeseries, lookback=1, normalized=False, _max=None, _min=None
):
    if not normalized:
        if _max is None:
            _max = _timeseries.max()
        if _min is None:
            _min = _timeseries.min()

        timeseries = normalize(_timeseries, _max, _min)
    else:
        timeseries = _timeseries
    # divide the time series in input instances of X: #lookback steps y: #lookback+1-th value of the time series
    # so we acquire #df * (complexity(df)-lookback)
    X = None
    y = None
    for i in range(lookback, len(timeseries)):
        X_i = np.asarray(timeseries[i - lookback : i]).reshape(
            (1, len(timeseries[i - lookback : i]), 1)
        )
        X = np.concatenate((X, X_i)) if X is not None else X_i
        y_i = np.asarray(timeseries[i]).reshape((-1, 1))
        y = np.concatenate((y, y_i)) if y is not None else y_i

    return X, y, _max, _min
