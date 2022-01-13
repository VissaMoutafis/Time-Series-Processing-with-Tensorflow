import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping

from src.utilities import *


class TimeSeriesForecastModel:
    """ 
        This is the model wrapper for timeseries forecasting.
        @input_dim : input dimension of the timeseries segments, equal to window_size or 'look-back'
        @lstm_units : list of number of units per lstm layer
        @dropout : dropout rate (float) or None if no dropout needed
        @_optimizer : the optimizer used to fit the model, default is adam
        @_loss : the loss function of the model, default is mse
    """
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
                self.model.add(layers.LSTM(units=u, return_sequences=True, input_shape=self.input_dim, dropout= 0 if self.dropout is None else self.dropout))
            elif i == len(self.lstm_units) - 1:
                self.model.add(layers.LSTM(units=u))
            else:
                self.model.add(layers.LSTM(units=u, return_sequences=True, dropout= 0 if self.dropout is None else self.dropout))

        # final output layer
        self.model.add(layers.Dense(input_dim[-1]))

        self.model.compile(optimizer=self.optimizer, loss=self.loss)


    """ 
        Wrapper for model fitting given the @X, @y data and labels the @epochs (default 100) and the @batch_size (default 128)
    """
    def fit(self, X, y, epochs=100, batch_size=128):
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
        self.D_train = X_train, y_train
        self.D_test = X_test, y_test
        es = EarlyStopping(monitor="val_loss", mode="min", patience=15, min_delta=1e-4)
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
        return self.model.predict(X, batch_size=64)

    def evaluate(self, X, y_true):
        """The evaluate function will print a graph of all"""
        return self.model.evaluate(X, y_true, batch_size=64)


class TimeSeriesForecast:
    """ 
        Framework to solve the timeseries prediction problem.
        @model: the model that we will use to solve the model. One easy to use model is the TimeSeriesForecastModel class.
        @dataset: the timeseries dataset of size [N x C], N = #timeseries, C = complexity of each timeseries
        @timeseries_labels: labels of each one of them for later plotting, default will be a numeric value index 
    """
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

    """ 
        Routine to solve the problem of forecasting:
        for each timeseries in the dataset
        1. preprocess the timeseries by scalling it and segment it in [C-@lookback, @lookback, 1] samples
        2. divide it to train and test segments 
        3. concat the 2 parts into the respective storage to keep for later use
        
        At the end fit the model to solve the problem
        
        @lookback: factor of looking back while dividing a given timeseries
        @epochs: eopchs to train the model
        @batch_size: batch size hyper parameter of model fitting 
    """
    def solve(self, lookback=4, epochs=100, batch_size=128):
        self.trained = True
        
        for timeseries in self.dataset[:]:
            # preprocess timeseries
            X_normalized, y_normalized, _max, _min = preprocess_timeseries(timeseries, lookback)
            self.max.append(_max)
            self.min.append(_min)

            # train-test for timeseries
            X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, shuffle=False)
            
            # concat the total training dataset
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

        # train the model given all training data
        self.history = self.model.fit(self.X_train_all, self.y_train_all, epochs, batch_size)


    """ 
        Function to plot all the true and predicted timeseries for the given ids in the @timeseries_idx
    """
    def plot_graphs(self, timeseries_idx=None):
        if not self.trained:
            raise Exception("Problem not solved")
        
        # default indexing-labeling
        if timeseries_idx is None:
            timeseries_idx = range(len(self.labels))

        # plot and legend init
        figure(figsize=(20, 10))
        lgnd = []
        
        for i in timeseries_idx:
            # print a msg to the user and change the plot's legend
            print(f'Ploting pred for {self.labels[i]}')
            lgnd += [f"{self.labels[i]}-True", f"{self.labels[i]}-Pred"]
            
            # get the respective eval data and plot the y_true 
            X, y = self.eval_data[i]
            plt.plot(
                range(len(y.reshape(-1))),
                reverse_normalize(y, self.max[i], self.min[i]).reshape(-1),
                "o-",
            )
            
            # get the predicted timeseries for the eval data and print it as well
            y_pred = self.model.predict(X)
            plt.plot(
                range(len(y_pred.reshape(-1))),
                reverse_normalize(y_pred, self.max[i], self.min[i]).reshape(-1),
                "o-",
            )
        
        # print legend and show the plot
        plt.legend(lgnd)
        plt.show()
