# -*- coding: utf-8 -*-
"""
Writters:
  Aristi Papstavrou
  Vissarion Moutafis


Time Series Forecasting using tensorflow lstm models.

usage: 
  ~$ python detect.py [-h] [-d DATASET_PATH] [-n N_SAMPLES] [-mae ERROR_VALUE]
"""

import pandas as pd
from src.ArgParser import *
from src.TimeSeriesAutoEncoderFramework import *
from config.detect_config import *


def main():
    # create the needed parser
    cmd_args = create_hyperparameter_parser(2)
    n_samples = int(cmd_args.n_samples)
    mae = float(cmd_args.mae)
    input_dataset_path = cmd_args.dataset_path

    # get the timeseries sample in a pandas dataframe
    timeseries_df = (
        pd.read_csv(input_dataset_path, sep="\t", index_col=0, header=None)
        .astype(np.float32)
        .sample(n_samples)
    )
    # get the indices in a list
    TIME_SERIES_IDS = timeseries_df.index.tolist()
    # create the timeseries autoencoder model
    model1 = LSTMAutoEncoder((LOOKBACK, 1), LSTM_LAYERS,timeseries_df.to_numpy(),batch_size = BATCH_SIZE, dropout=DROPOUT_RATE)
    # initiate adequate framework
    model1.solve(lookback=LOOKBACK, epochs=EPOCHS, batch_size = BATCH_SIZE)
    
    
    X_train_pred = model1.model.predict(model1.X_train_all)
    train_mae_loss = np.mean(np.abs(X_train_pred - model1.X_train_all), axis=1)
    sns.displot(train_mae_loss, bins=50, kde=True)
    plt.show()
    
    
    plot_anomalies(model1,TIME_SERIES_IDS,mae = mae)
    
    plt.plot(model1.history.history['loss'], label='train')
    plt.plot(model1.history.history['val_loss'], label='test')
    plt.legend()
    plt.show();


if __name__ == "__main__":
    main()
