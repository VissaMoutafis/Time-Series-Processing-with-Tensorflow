# -*- coding: utf-8 -*-
"""
Writters:
  Aristi Papstavrou
  Vissarion Moutafis


Time Series Forecasting using tensorflow lstm models.

usage: 
  ~$ python forecast.py [-h] [-d DATASET_PATH] [-n N_SAMPLES]
"""
import pandas as pd
from src.ArgParser import *
from src.TimeSeriesForecastingFramework import *
from config.forecast_config import *


def main():
    # create the needed parser
    cmd_args = create_hyperparameter_parser(1)
    n_samples = int(cmd_args.n_samples)
    input_dataset_path = cmd_args.dataset_path

    # get the timeseries sample in a pandas dataframe
    timeseries_df = pd.read_csv(input_dataset_path, sep="\t", index_col=0, header=None).astype(np.float32).sample(n_samples)
        
    # get the indices in a list
    TIME_SERIES_ID = timeseries_df.index.tolist()

    # create the timeseries prediction model
    model = TimeSeriesForecastModel((LOOKBACK, 1), LSTM_LAYERS, dropout=DROPOUT_RATE, _loss=LOSS)

    # initiate a timeseries forcasting framework
    problem = TimeSeriesForecast(model, timeseries_df.to_numpy(), TIME_SERIES_ID)

    # solve the problem
    problem.solve(lookback=LOOKBACK, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # plot graphs based on index of timeseries (Comment out if you dont care)
    problem.plot_graphs()


if __name__ == "__main__":
    main()
