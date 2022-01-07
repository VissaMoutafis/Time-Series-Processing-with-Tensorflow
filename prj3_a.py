# -*- coding: utf-8 -*-
"""
Writters:
  Aristi Papstavrou
  Vissarion Moutafis


Time Series Forecasting using tensorflow lstm models.

usage: 
  ~$ python prj3_a.py [-h] [-d DATASET_PATH] [-n N_SAMPLES]
"""
import pandas as pd
from ArgParser import *
from TimeSeriesForecastingFramework import *
from ex1_config import *


def main():
    # create the needed parser
    cmd_args = create_Parser(1)
    n_samples = int(cmd_args.n_samples)
    input_dataset_path = cmd_args.dataset_path
    timeseries_df = (
        pd.read_csv(input_dataset_path, sep="\t", index_col=0, header=None)
        .astype(np.float32)
        .sample(n_samples)
    )
    TIME_SERIES_ID = timeseries_df.index.tolist()
    model = TimeSeriesForecastModel((LOOKBACK, 1), LSTM_LAYERS, dropout=DROPOUT_RATE)
    problem = TimeSeriesForecast(model, timeseries_df.to_numpy(), TIME_SERIES_ID)
    problem.solve(lookback=LOOKBACK, epochs=EPOCHS)
    problem.plot_graphs()


if __name__ == "__main__":
    main()
