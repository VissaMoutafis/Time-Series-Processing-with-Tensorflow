# -*- coding: utf-8 -*-
"""
Writters:
  Aristi Papstavrou
  Vissarion Moutafis


Time Series Forecasting using tensorflow lstm models.

usage: 
  ~$ python reduce.py [-h] [-d DATASET_PATH] [-q QUERY_PATH] [-od OUTPUT_DATASET_PATH] [-q QUERY_PATH] 
"""
import pandas as pd
from scipy.sparse import data
from src.ArgParser import *
from src.TimeSeriesCNNAutoEncoderFramework import *
from config.reduce_config import *


def main():
    # create the needed parser
    cmd_args = create_hyperparameter_parser(3)
    n_samples = N_TRAINING_SAMPLES
    input_dataset_path = cmd_args.dataset_path
    query_dataset_path = cmd_args.query_set
    output_query = cmd_args.output_query_path
    output_input = cmd_args.output_path

    # get the timeseries sample in a pandas dataframe
    timeseries_input_df = (
        pd.read_csv(input_dataset_path, sep="\t", index_col=0, header=None)
        .astype(np.float32)
    )
    timeseries_query_df = (
        pd.read_csv(query_dataset_path, sep="\t", index_col=0, header=None)
        .astype(np.float32)
    )
    # get the indices in a list
    train_timeseries_dataset = timeseries_input_df.sample(n_samples)
    TIME_SERIES_TRAIN_ID = train_timeseries_dataset.index.tolist()

    TIME_SERIES_INPUT_ID = timeseries_input_df.index.tolist()
    TIME_SERIES_QUERY_ID = timeseries_query_df.index.tolist()

    # create the timeseries prediction model
    autoencoder = TimeSeriesComplexityReducerModel(
        LOOKBACK, 
        CNN_LAYER_SETTINGS, 
        latent_dim=LATENT_DIM,
        pool_size=POOL_SIZE, 
        dropout_rate=DROPOUT_RATE,
        _loss=LOSS, 
        verbose=True)

    # create the problem statement
    problem=TimeSeriesComplexityReducer(autoencoder, train_timeseries_dataset.to_numpy(), TIME_SERIES_TRAIN_ID)
    
    # solve the problem
    problem.solve(epochs=EPOCHS, batch_size=BATCH_SIZE)
    problem.plot_graphs([0])
    
##############################################################################################################################       
    
    #create compressed input data
    problem.create_compressed_file(timeseries_input_df.to_numpy(),TIME_SERIES_INPUT_ID, output_input)
    
    #create compressed query file
    problem.create_compressed_file(timeseries_query_df.to_numpy(), TIME_SERIES_QUERY_ID, output_query)
    
if __name__ == "__main__":
    main()
