import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping

from utilities import *


class TimeSeriesForecastModel:
    def __init__(
        self, input_dim, lstm_units, dropout=None, _optimizer="adam", _loss="mse"
    ):
        self.history = None
        self.D_train = None
        self.D_test = None
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.optimizer = _optimizer
        self.loss = _loss
        self.input_dim = input_dim

        # create the model architecture
        self.model = models.Sequential()
        for i, u in enumerate(self.lstm_units):
            if i == 0:
                self.model.add(
                    layers.LSTM(
                        units=u, return_sequences=True, input_shape=self.input_dim
                    )
                )
            elif i == len(self.lstm_units) - 1:
                self.model.add(layers.LSTM(units=u))
            else:
                self.model.add(layers.LSTM(units=u, return_sequences=True))
            if self.dropout is not None:
                self.model.add(layers.Dropout(self.dropout))

        # final output layer
        self.model.add(layers.Dense(input_dim[-1]))

        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self, X, y, epochs=100, batch_size=128):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        self.D_train = X_train, y_train
        self.D_test = X_test, y_test
        es = EarlyStopping(monitor="val_loss", mode="min", patience=10, min_delta=1e-4)
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[es],
        )
        return self.history

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y_true):
        """The evaluate function will print a graph of all"""
        return self.model.evaluate(X, y_true, batch_size=64)


class TimeSeriesForecast:
    def __init__(self, model, dataset, timeseries_labels=None):
        self.dataset = dataset
        self.model = model
        self.history = []
        self.trained = False
        self.eval_data = []
        self.labels = timeseries_labels if timeseries_labels is not None else list(range(len(dataset)))
        self.max = []
        self.min = []
        self.X_train_all = None
        self.y_train_all = None

    def solve(self, lookback=4, epochs=100, batch_size=128):
        self.trained = True
        
        for timeseries in self.dataset:
            # preprocess timeseries
            X_normalized, y_normalized, _max, _min = preprocess_timeseries(
                timeseries, lookback
            )
            self.max.append(_max)
            self.min.append(_min)

            # train-test for timeseries
            train_lim = 5*len(X_normalized)//6
            X_train, X_test, y_train, y_test = X_normalized[:train_lim], X_normalized[train_lim:], y_normalized[:train_lim], y_normalized[train_lim:]
            
            self.X_train_all = (
                X_train
                if self.X_train_all is None
                else np.concatenate((self.X_train_all, X_train))
            )
            self.y_train_all = (
                y_train
                if self.y_train_all is None
                else np.concatenate((self.y_train_all, y_train))
            )

            # add the eval data in the respective array
            self.eval_data.append((X_test, y_test))

        self.history = self.model.fit(
            self.X_train_all, self.y_train_all, epochs, batch_size
        )

    def plot_graphs(self, timeseries_idx=None):
        if not self.trained:
            raise Exception("Problem not solved")
        if timeseries_idx is None:
            timeseries_idx = range(len(self.labels))

        figure(figsize=(20, 10))
        lgnd = []
        for i in timeseries_idx:
            print(f'Ploting pred for {self.labels[i]}')
            lgnd += [f"{self.labels[i]}-True", f"{self.labels[i]}-Pred"]
            X, y = self.eval_data[i]
            plt.plot(
                range(len(y.reshape(-1))),
                reverse_normalize(y, self.max[i], self.min[i]).reshape(-1),
                "o-",
            )
            y_pred = self.model.predict(X)
            plt.plot(
                range(len(y_pred.reshape(-1))),
                reverse_normalize(y_pred, self.max[i], self.min[i]).reshape(-1),
                "o-",
            )

        plt.legend(lgnd)
        plt.show()
