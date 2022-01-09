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
    
def standardize(X, mean, sigma):
    return (X - mean) / sigma

def reverse_standardize(X, mean, sigma):
    return X * sigma + mean

def create_dataset(_timeseries, time_steps=1, standardized=False, mean = None, sigma = None):
    if not standardized:
      if mean is None:
        mean = _timeseries.mean()
      if sigma is None:
        sigma = _timeseries.std()

      timeseries = standardize(_timeseries, mean, sigma)
    else:
      timeseries = _timeseries
    
    Xs = None
    ys = None
    for i in range(time_steps, len(timeseries)):
        X_i = np.asarray(timeseries[i - time_steps : i]).reshape((1, len(timeseries[i - time_steps : i]), 1))
        Xs = np.concatenate((Xs, X_i)) if Xs is not None else X_i
        y_i = np.asarray(timeseries[i]).reshape((-1, 1))
        ys = np.concatenate((ys, y_i)) if ys is not None else y_i


    return Xs, ys, mean, sigma
