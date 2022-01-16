import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping
from keras.layers.merge import concatenate

from math import ceil
from numpy import savetxt

from src.utilities import *


class TimeSeriesComplexityReducerModel():
  n_conv_filt_default = 10
  """ 
    Model Wrapper for the Complexity reduction problem. Creates the model and init the necessary variables to function.
    
    @window_size: the look-back value
    @conv_layers_settings: a list of dicts that will provide the conv layers with filters and kernel parameters
    @latent_dim: the dimension that we will project all the [@window_size, 1] vectors
    @pool_size: size of pooling filters 
    @_optimizer: the optimizer we use to solve the problem
    @_loss: the loss function that we trying to fit
    @dropout_rate: the dropout rate between layers
    @verbose: verbosity flag
  """
  def __init__(self, window_size, conv_layers_setting=[], latent_dim=3, pool_size=2, _optimizer='adam', _loss='bce', dropout_rate=None, verbose=False, trained_model_path=None):
    self.verbose=verbose
    self.input_dim = window_size
    self.latent_dim = latent_dim 
    self.pool_size = pool_size
    
    self.history = None
    self.D_train = None
    self.D_test = None
    self.dropout = dropout_rate
    self.optimizer = _optimizer
    self.loss = _loss

    if trained_model_path is None:
      # init input layer
      input_w = layers.Input(shape=(window_size,1))
      input_dim = window_size
      x = input_w

      # add the convolution layers
      for conv_settings in conv_layers_setting:
        # set up convolution filters and kernel dimensions 
        filters = self.n_conv_filt_default
        if 'filters' in conv_settings:
          filters = conv_settings['filters']
        kernel_size = conv_settings['kernel_size']

        # add the layer into the encoder
        x = layers.Conv1D(filters, kernel_size, padding="same", activation="relu")(x)
        
        # downsample if you can
        if ceil(input_dim/pool_size) > latent_dim:
          input_dim = ceil(input_dim/pool_size)
          x = layers.MaxPooling1D(pool_size, padding="same")(x)

        if self.dropout is not None:
          x = layers.Dropout(self.dropout)(x)

      # final compression with dense layer and a relu activation
      x = layers.Flatten()(x)
      x = layers.Dense(latent_dim, activation='relu')(x)
      encoded = layers.Reshape((latent_dim, 1))(x)
      
      # create the encoder model
      self.encoder = models.Model(input_w, encoded, name='encoder')
      if self.verbose:
        self.encoder.summary()  
      self.encoder.compile(optimizer=_optimizer, loss=_loss)
    else:
      self.encoder = models.load_model(trained_model_path)
      

    if trained_model_path is None:
      # decoder model architecture
      output_dim = latent_dim 
      
      # use transpose convolutions to recreate the timeseries
      x = layers.Conv1DTranspose(1, latent_dim, activation='relu', padding="same")(encoded)
      
      # up sampling
      if output_dim*pool_size <= window_size:
          output_dim = output_dim*pool_size
          x = layers.UpSampling1D(pool_size)(x)
      
      # add dropout
      if self.dropout is not None:
        x = layers.Dropout(self.dropout)(x)

      # for all conv layers provided add a reconstructor layer of the input timeseries
      for i, conv_settings in enumerate(conv_layers_setting[::-1]):
        # parse the semantics from the layer setting given by the user
        padding="same"
        filters = self.n_conv_filt_default
        if 'filters' in conv_settings:
          filters = conv_settings['filters']
        kernel_size = conv_settings['kernel_size']

        # add the transpose convolution
        x = layers.Conv1DTranspose(filters, kernel_size, activation="relu", padding=padding)(x)
        
        # upscalling if we haven't reached the input dimension just yet      
        if output_dim*pool_size <= window_size:
          output_dim = output_dim*pool_size
          x = layers.UpSampling1D(pool_size)(x)
        
        if self.dropout is not None:
          x = layers.Dropout(self.dropout)(x)

      # Final activation layer that we use sigmoid activation because we use man mix scalling
      x = layers.Flatten()(x)
      x = layers.Dense(window_size, activation='sigmoid')(x)
      decoded = layers.Reshape((window_size, 1))(x)

      self.autoencoder = models.Model(input_w, decoded, name="autoencoder")
      if self.verbose:
        self.autoencoder.summary()
      
      self.autoencoder.compile(optimizer=_optimizer, loss=_loss)

  """ 
    fit the model to the given @X, @y, for @epochs iteration over the dataset, dividing it by @batches subsets. Also applies early training 
    and returns the history after fitting.
  """
  def fit(self, X, y, epochs=50, batch_size=128):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    self.D_train = X_train, y_train
    self.D_test = X_test, y_test
    es = EarlyStopping(monitor="val_loss", mode="min", patience=10, min_delta=1e-4)
    self.history = self.autoencoder.fit(
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
    return self.autoencoder.predict(X)

  """
    Use the encoder model to reduce the dimensionality of the given input @X
  """
  def encode(self, X):
    return self.encoder.predict(X)

  def evaluate(self, X, y_true):
    return self.autoencoder.evaluate(X, y_true, batch_size=64)

  def save_solver(self, path):
    self.encoder.save(path)
  
class TimeSeriesComplexityReducer:
  """ 
    Framework to solve the Complexity Reduction problem for timeseries.
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
    Use the provided model to fit the problem for given @epochs and @batch_size. 
    Divide the timeseries into segments and construct a total training dataset to train the model.
    Keep some of the data per timeseries for evaluation later on.
  """
  def solve(self, epochs=100, batch_size=128):
    self.trained = True
        
    for timeseries in self.dataset:
      # preprocess timeseries
      X_normalized, y_normalized, _max, _min = preprocess_timeseries(timeseries, self.model.input_dim, _for_='dim_reduction')
      self.max.append(_max)
      self.min.append(_min)

      # train-test for timeseries
      X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, shuffle=False)
            
      self.X_train_all = X_train if self.X_train_all is None else np.concatenate((self.X_train_all, X_train))
      
      self.y_train_all = y_train if self.y_train_all is None else np.concatenate((self.y_train_all, y_train))
      

      # add the eval data in the respective array
      self.eval_data.append((X_test, y_test))
    
    self.history = self.model.fit(self.X_train_all, self.y_train_all, epochs=epochs, batch_size=batch_size)

  """
    Graph plotting for the visualization of the autoencoder functionality and performance 
  """
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
      y_pred = self.model.predict(X)

      y = np.concatenate((y[:, 0, 0].reshape(-1),y[-1, :, 0].reshape(-1)))
      y_pred = np.concatenate((y_pred[:, 0, 0].reshape(-1),y_pred[-1, :, 0].reshape(-1)))

      plt.plot(
      range(len(y.reshape(-1))),
      reverse_normalize(y, self.max[i], self.min[i]).reshape(-1),
      "o-",
      )

      plt.plot(
      range(len(y_pred.reshape(-1))),
      reverse_normalize(y_pred, self.max[i], self.min[i]).reshape(-1),
      "o-",
      )
    plt.legend(lgnd)
    plt.show()

  """ 
    Divide the @_timeseries to segments of @window size with step of @window
  """
  def sample_timeseries(self, _timeseries, window):
    _max = _timeseries.min()
    _min = _timeseries.max()
    timeseries = normalize(_timeseries, _max, _min)
  
    X = None
    for i in range(window, len(timeseries), window):
      X_i = np.asarray(timeseries[i-window:i]).reshape((1, len(timeseries[i-window:i]), 1))
      X = np.concatenate((X, X_i)) if X is not None else X_i
    
    if len(timeseries) % window > 0:
      X_i = np.asarray(timeseries[-1-window:-1]).reshape((1, len(timeseries[-1-window:-1]), 1))
      X = np.concatenate((X, X_i))

    return X, X, _max, _min


  """
    Sample the timeseries and reduce the dimension
    @timeseries_ndarray : [N x C] array with N timeseries of complexity C
    @TIMESERIES_IDS: labels for timeseries
  """
  def reduce_and_export(self, timeseries_ndarray, TIMESERIES_IDS):
    # helper to reduce the complexity of a timeseries
    def _reduce(timeseries):
      X, y, _max, _min = self.sample_timeseries(timeseries, self.model.input_dim) 
      return self.model.encode(X).reshape(-1)
    
    to_export = np.apply_along_axis(_reduce, 1, timeseries_ndarray)
    to_export = np.round(to_export,2)
    id_nums = timeseries_ndarray.shape[0]
    
    df_export = pd.DataFrame(data=to_export)
    
    #add ids to dataframe
    df_export['id'] = TIMESERIES_IDS[:id_nums]
    
    # shift column 'id' to first position
    first_column = df_export.pop('id')
    
    # insert column using insert(position,column_name,first_column) function
    df_export.insert(0, 'id', first_column)
        
    return df_export

  def create_compressed_file(self,timeseries_ndarray,TIMESERIES_IDS,out_filename='test.out'):
    df_to_export = self.reduce_and_export(timeseries_ndarray,TIMESERIES_IDS)
    df_to_export.to_csv(out_filename,sep='\t',line_terminator='\r\n',header=False, index=False)
