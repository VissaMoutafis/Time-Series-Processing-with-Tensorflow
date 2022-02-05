# Time-Series-Processing-with-Tensorflow

In this project we will explore 3 different tasks on Timeseries, using the nasdaq stocks dataset from 2007 to 2017.

## Instalation
To download the project just `git clone` the repository.
To install prerequisities:
```bash
cd Time-Series-Processing-with-Tensorflow
pip3 install requirements.txt
```

In the root directory there are 3 python files that implement the tasks at hand. Each file gets specific command line arguments and all of the programs will use 3 general arguments to help with their functionality:
- `-h` for printing the 'help' message
- `--train` for setting the train mode on
- `--model-path` in case we want to save/load the model in/from some other path, instead of the root directory

## Configuration Files
Each and every program will get its own `*_cofig.py` file that will contain, definitiions for parameters. The user is welcome to expreriment and change the default values of the paremeters. Those parameteers are used during training and visualization routines. 


## Time Series Forecasting with LSTMs
In order to use the forecast routine:
```bash
python3 forecast.py -n [nom of selected timeseries] -d [dataset path] [--train] [--model-path] [-h]
```
For forcasting we will get $n$ timeseries and divide it in continuous sub-series. Then we will split this batch in train and test, train the model - if train mode is enabled - and plot the true vs predicted graph to visualize our forecasting results. For further insight check README pdf file and the comments in the respective source file.

## Outlier Detection using LSTM Autoencoder
```bash
python3 detect.py -n [num of selected timeseries] -d [dataset path] -mae [MAE threshold] [--train] [--model-path] [-h]
```
For outlier detection we will follow the same split and train routine as before with the exception that we will use different predition evaluation metrics. The `-mae` argument will be usefull to print the respective outliers in the given test timeseries. For further explanation check README pdf file and comments in the respective source file.

## Dimensionality Reduction using CNNs Autoencoder
```bash
python3 reduce.py -d [dataset path] -q [query dataset path] -od [output datset path] -oq [output query path] [--train] [--model-path] [-h]
```
In order to reduce the dimensionality we will train another autoencoder, but we will keep the encoder part and according to the given datasets we will use the encoder to reduce the complexity of the timeseries at hand. For further explanation check README pdf file and comments in the respective source file.

## Miscellaneous
We provide couple of runs' results (`misc/Metrics/`), as well as the models we used to acquire those results (`misc/models/`). We also provide a detailed documentation in Greek (english documentation will be provided in the future).
The test results for dimensionality reduction was used in couple of runs for [this]() Project, where we applied kNN and clustering algorithm onto the reduced dataset.
## Collaborators
Vissarion Moutafis - [VissaMoutafis](https://github.com/VissaMoutafis)
Aristi Papastavrou - [AristiPap](https://github.com/AristiPap)
