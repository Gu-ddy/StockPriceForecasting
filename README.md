# Stock price forecasting: a comparison between classical statistical models (ARIMA) and recurrent neural networks (LSTM)

## Description

![Comparison of the third RNN and ARIMA on horizon 10 predictions](https://github.com/Gu-ddy/StockPriceForecasting/blob/main/Images/Comparison.png)  

The ultimate objective of this project is to compare *ARIMA* (AutoRegressive Integrated Moving Average) models and *LSTM* based recurrent neural networks, in a widely known task: predicting a financial asset.

Specifically, the subject of prediction will be the closing values of the *FTSE MIB* (Financial Times Stock Exchange Milano), the most representative stock index of the Italian financial market. The observations used for analysis span a time period of 10 years, starting from November 3, 2011, to November 3, 2021.

The improvement in the performance of a neural network considering additional data has also been analyzed. Such data includes transaction volume on the index stocks, as well as variations between openings and closings. For the same reason, the closing values of the german *DAX* (Deutscher Aktien Index) and the openings of the american *S&P 500* (Standard & Poor 500) have also been utilized.

Likewise, the impact of time series differentiation (which can be considered as a sort of standardization) has also revealed a key factor for the network.

## Results

For evaluation, both a 10-days and a 1-day horizon predictions have been considered. The dataset has been divided into two parts. Approximately 80% of the observations, specifically the values from November 3, 2011, to November 4, 2019, were used to estimate the coefficients of various models. The subsequent observations, from November 5, 2019, to November 3, 2021, were reserved for error calculation and thus performance evaluation.

*Mean absolute error* (lower is better) was used for evaluation of both the models and training of the network.  *AIC* (Akaike information Criterion) was used instead for identification and parameter estimation of the ARIMA model.

| Models Horizon 10                           | MAE      |
|---------------------------------------------|----------|
| RNN no external and not differentiated data | 1006.998 |
| RNN no external differentiated data         | 443.432  |
| **RNN external and differentiated data**        | **440.530**   |
| ARIMA                                       | 578.280  |


| Models Horizon 1                     | MAE     |
|--------------------------------------|---------|
| **RNN external and differentiated data** | **12.177**  |
| ARIMA                                | 309.902 |

We can draw the following conclusions:

1. Differentiation has turned out to be a fundamental pre-processing step for convergence of neural networks.
2. Neural network models outperform ARIMA given a differentiated input.
3.  The gap between the two is larger for a smaller horizon prediction.
4. Using external data slightly improves the performances of RNNs.
## Architecture details
The model was built and trained entirely using Keras functional API.

A summary diagram of the architecture is given by 

![Model architecture diagram](https://github.com/Gu-ddy/StockPriceForecasting/blob/main/Images/model.png)
On every LSTM layer l2 regularizers were used to make the network more robust. Likewise a dropout layer was added before the last fully connected layer.

The used optimizer was Adam with default learning rate and batch size equal to 256. 


## Datasets
All used data is available inside the dedicated folder, nonetheless, cleaning and simple feature engineering operations were performed before conducting the analysis, for details look at the main notebook.

## Usage
For ready-to-go usage and to reproduce the results, you can simply run the main notebook installing the dependencies by:
<code> 
		pip install -r requirements.txt	
<code>

## Future work
There are several improvements that could be carried out, among them we have:
* Including fundamental analysis (only for the network).
* Using a more sophisticated evaluation framework.
* Include exogenous data also for ARIMA.
* Improving the optimization pipeline (e.g. adding early stopping with a validation set).
* Including transformers in the architecture.

