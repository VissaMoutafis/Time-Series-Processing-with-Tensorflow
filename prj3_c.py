# -*- coding: utf-8 -*-
"""
Writters:
  Aristi Papstavrou
  Vissarion Moutafis


Time Series Forecasting using tensorflow lstm models.

usage: 
  ~$ python prj3_c.py [-h] [-d DATASET_PATH] [-q QUERY_PATH] [-od OUTPUT_DATASET_PATH] [-q QUERY_PATH] 
"""
import pandas as pd
from scipy.sparse import data
from ArgParser import *
from TimeSeriesCNNAutoEncoderFramework import *
from ex1_config import *


def main():
    # create the needed parser
    cmd_args = create_hyperparameter_parser(3)
    n_samples = 10 #int(cmd_args.n_samples)
    input_dataset_path = cmd_args.dataset_path
    query_dataset_path = cmd_args.query_set
    output_query = cmd_args.output_query_path
    output_input = cmd_args.output_path

    # get the timeseries sample in a pandas dataframe
    timeseries_input_df = (
        pd.read_csv(input_dataset_path, sep="\t", index_col=0, header=None)
        .astype(np.float32)
        .sample(n_samples)
    )
    timeseries_query_df = (
        pd.read_csv(query_dataset_path, sep="\t", index_col=0, header=None)
        .astype(np.float32)
        .sample(n_samples)
    )
    # get the indices in a list
    TIME_SERIES_INPUT_ID = timeseries_input_df.index.tolist()
    TIME_SERIES_QUERY_ID = timeseries_query_df.index.tolist()
    ###########################################CONVERSION OF INPUT FILE###############################################
    window_length = 10
    X_all,_, _max, _min = preprocess_timeseries(timeseries_input_df.to_numpy()[0], window_length)
    X_train, X_test = train_test_split(X_all, test_size=0.33, shuffle=False)

    # create the timeseries prediction model
    autoencoder = TimeSeriesComplexityReducerModel(window_length, [{'filters':64, 'kernel_size':7}, {'filters':64, 'kernel_size':7}], latent_dim=7, pool_size=2)
    test_samples = 50

    plot_examples(reverse_normalize(X_all, _max, _min), reverse_normalize(autoencoder.predict(X_all), _max, _min))

    problem=TimeSeriesComplexityReducer(autoencoder, timeseries_input_df.to_numpy(), timeseries_input_df)
    
    # solve the problem
    problem.solve(epochs=EPOCHS)

    # plot graphs based on index of timeseries
    #problem.plot_graphs()
    
    ########################################CONVERSION OF QUERY FILE #################################################
    X_all,_, _max, _min = preprocess_timeseries(timeseries_query_df.to_numpy()[0], window_length)
    X_train, X_test = train_test_split(X_all, test_size=0.33, shuffle=False)

    # create the timeseries prediction model
    autoencoder2 = TimeSeriesComplexityReducerModel(window_length, [{'filters':64, 'kernel_size':7}, {'filters':64, 'kernel_size':7}], latent_dim=7, pool_size=2)
    test_samples = 50

    plot_examples(reverse_normalize(X_all, _max, _min), reverse_normalize(autoencoder2.predict(X_all), _max, _min))

    problem2=TimeSeriesComplexityReducer(autoencoder2, timeseries_query_df.to_numpy(), timeseries_query_df)
    
    # solve the problem
    problem2.solve(epochs=EPOCHS)

    # plot graphs based on index of timeseries
    #problem2.plot_graphs()   
    
##############################################################################################################################       
    #preparation for question d)
    
    #create compressed input data
    problem.create_compressed_file(timeseries_input_df.to_numpy(),TIME_SERIES_INPUT_ID,'input_data.csv')
    
    #create compressed query file
    problem2.create_compressed_file(timeseries_query_df.to_numpy(),TIME_SERIES_QUERY_ID,'query_data.csv')
    
if __name__ == "__main__":
    main()
