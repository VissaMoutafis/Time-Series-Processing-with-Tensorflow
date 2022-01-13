# -*- coding: utf-8 -*-
"""
Writters:
  Aristi Papstavrou
  Vissarion Moutafis


Time Series AutoEncoder using tensorflow lstm models.

usage: 
  ~$ python prj3_a.py [-h] [-d DATASET_PATH] [-n N_SAMPLES]
"""

import argparse
import sys

sys.path.append('..')
import numpy as np
import pandas as pd

from keras import layers, optimizers, losses, metrics
from keras.models import Sequential
from keras import Model
from keras.models import load_model

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import BatchNormalization
from keras import Input
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from matplotlib import pyplot as plt

import tensorflow as tf
np.random.seed(1)

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping

from utilities import *
# input_dataset_path = 'drive/MyDrive/Project-Datasets/nasd_input.csv'
# query_dataset_path = 'drive/MyDrive/Project-Datasets/nasd_query.csv'
# DATASET_SIZE = 3
# LOOKBACK = 20

# timeseries_df = pd.read_csv(input_dataset_path, sep='\t', index_col=0, header=None).astype(np.float32).sample(DATASET_SIZE)
# TIME_SERIES_ID = timeseries_df.index.tolist()


class LSTMAutoEncoder():
    def __init__(self, input_dim, lstm_units, dataset, batch_size = 128, dropout=None, _optimizer='adam', _loss='mse'):
        super(LSTMAutoEncoder, self).__init__()
        self.D_train = None
        self.D_test = None
        self.lstm_units = lstm_units
        self.batch_size =  batch_size
        self.dropout = dropout
        self.optimizer = _optimizer
        self.loss = _loss
        self.input_dim = input_dim
        self.models = {}
        self.input_dim = input_dim      
        self.history = {}
        self.trained = False
        self.eval_data = []
        self.dataset = dataset
        self.mean = []
        self.sigma = []
        self.X_train_all = None
        self.y_train_all = None
        
        self.model = models.Sequential()
        #encoder loop
        _batch_size = self.batch_size
        for i, u in enumerate(self.lstm_units):
          self.model.add(BatchNormalization())
          if i == 0: 
            encoded = layers.LSTM(units=u, return_sequences=True, input_shape=self.input_dim, batch_input_shape=(_batch_size, input_dim[0], input_dim[-1]),dropout=self.dropout)
            self.model.add(encoded)

          elif i == len(self.lstm_units) - 1:
            encoded = layers.LSTM(units=u,batch_input_shape=(_batch_size, input_dim[0], input_dim[-1]),dropout=self.dropout)
            self.model.add(encoded)
            self.decoded = layers.RepeatVector(input_dim[0])

          else: 
            self.model.add(layers.LSTM(units=u, return_sequences=True,batch_input_shape=(_batch_size, input_dim[0], input_dim[-1]),dropout=self.dropout))
          if self.dropout is not None:
            self.model.add(layers.Dropout(self.dropout))

        self.model.add(self.decoded)

        #decoder loop 
        for i, u in enumerate(self.lstm_units[::-1]):
          self.model.add(BatchNormalization())
          if i == len(self.lstm_units) - 1:
            encoded = layers.LSTM(units=u,return_sequences=True,batch_input_shape=(_batch_size, input_dim[0], input_dim[-1]),dropout=self.dropout)
            self.model.add(encoded)
          else: 
            self.model.add(layers.LSTM(units=u, return_sequences=True,batch_input_shape=(_batch_size, input_dim[0], input_dim[-1]),dropout=self.dropout))
          if self.dropout is not None:
            self.model.add(layers.Dropout(self.dropout))

        # final output layer
        self.model.add(layers.TimeDistributed(layers.Dense(units=input_dim[-1], activation="tanh")))
        
        self.model.compile(optimizer='adam', loss=self.loss)

    def fit(self, X, y, epochs=15, batch_size=128):
      _history = {}
      es = EarlyStopping(monitor="val_loss", mode="min", patience=10, min_delta=1e-4)
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
      _history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1,callbacks=[es],)
    
      return _history
    
    def solve(self, lookback=20, epochs=15, batch_size=128):
      self.trained = True

      for timeseries in self.dataset:
        # preprocess timeseries
        X_normalized, y_normalized, _mean, _sigma = create_dataset(timeseries,lookback)
        self.mean.append(_mean)
        self.sigma.append(_sigma)

        # train the model
        train_lim = 5*len(X_normalized)//6
        X_train, X_test, y_train, y_test = X_normalized[:train_lim], X_normalized[train_lim:], y_normalized[:train_lim], y_normalized[train_lim:]
          
        self.X_train_all = (
          X_train
          if self.X_train_all is None
          else np.concatenate((self.X_train_all,X_train))
        )
        self.y_train_all = (
          y_train
          if self.y_train_all is None
          else np.concatenate((self.y_train_all, y_train))
        )

        # add the eval data in the respective array
        self.eval_data.append((X_test,y_test))

      self.history = self.fit(self.X_train_all, self.y_train_all, epochs = epochs, batch_size = batch_size)

def plot_anomalies(model,TIME_SERIES_IDS,mae = 0.5):
  fig,axes = plt.subplots(1,len(TIME_SERIES_IDS),figsize = (30,10))
  timeseries_idx = range(len(TIME_SERIES_IDS))
  for i in timeseries_idx:
    X_test, y_test = model.eval_data[i]
    X_test_pred = model.model.predict(X_test)
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
    # in the last commit for threshold 3 we had double the anomalies. We used to have threshold 5. Now for 5 we only have 1 anomaly
    THRESHOLD = mae
    test_score_df = pd.DataFrame()
    test_score_df['loss'] = test_mae_loss.flatten()
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = test_score_df.loss > THRESHOLD
    test_score_df['close'] = X_test[:,1]
    #check if we are tresholding right
    #plt.plot(test_score_df.index, test_score_df.loss, label='loss')
    #plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
    #plt.xticks(rotation=25)
    #plt.legend();

    anomalies = test_score_df[test_score_df.anomaly == True]
    anomaly =  np.array(anomalies.close)

    test_score_df_close_new =  np.array(test_score_df.close)
    plt.rcParams['figure.figsize'] = [30, 10]
    axes[i].plot(
      test_score_df.index, 
      reverse_standardize(test_score_df_close_new,model.mean[i], model.sigma[i]).flatten(),
      label='stock real price'
    );
    axes[i].scatter(anomalies.index,
                    reverse_standardize(anomaly,model.mean[i], model.sigma[i]).flatten(),
                    color=sns.color_palette()[3],
                    s=52,
                    label='anomaly')
    axes[i].title.set_text('Stock Price vs Anomalies for timeseries:')
  plt.xticks(rotation=25)
  plt.legend()
  plt.show()