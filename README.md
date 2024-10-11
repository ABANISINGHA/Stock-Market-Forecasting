# Stock Market Forecasting Using ARIMA and LSTM

This repository focuses on predicting the Trading Volume of stocks using both classical time series models like ARIMA and deep learning models such as LSTM. We explore data preprocessing, model building, and compare the effectiveness of ARIMA and LSTM models on [stock market data](https://drive.google.com/file/d/13f2P7EvzXnwRQ8ps1OIOEbS-K16JLVOH/view?usp=sharing).

## Overview
Stock price forecasting is a critical aspect of trading strategies, risk management, and decision-making in the financial market. This project implements two types of models:

ARIMA (AutoRegressive Integrated Moving Average): A popular statistical model used for forecasting univariate time series data.


LSTM (Long Short-Term Memory): A type of recurrent neural network (RNN) capable of learning long-term dependencies and non-linear patterns in time series data.


The objective is to predict Trading Volume from historical stock data and compare the performance of ARIMA and LSTM models.

### Dataset

The dataset consists of historical stock data with the following columns:

- **Date**: The date of the record
- **Closing Price**: The price at market close
- **Opening Price**: The price at market open
- **High Price**: The highest price of the day
- **Low Price**: The lowest price of the day
- **Trading Volume**: The volume of stock traded
- **Rate of Change %**: Percentage change in stock price
- **Daily Price Change**, **Price Volatility**, **MA_5**, **MA_20**: Various stock indicators added by data processing
### Installation

#### Requirements:
- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `sklearn`, `tensorflow`, `pmdarima`, `statsmodels`

#### Install the required libraries:
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow pmdarima statsmodels
```

## Data Preprocessing

### 1. Handling Missing Data
Missing values can lead to inaccurate predictions. We use forward fill (`ffill()`) to handle missing data:

```python
stock_data.fillna(method='ffill', inplace=True)
```

### 2. Data Scaling
Before training the LSTM model, we scale the data to a range of [0, 1] using MinMaxScaler:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_trading_volume = scaler.fit_transform(stock_data['Trading Volume'].values.reshape(-1, 1))
```

### 3. Statistical Overview
A quick overview of the dataset's statistics:

```python
stock_data.describe()
```


This provides insights into the data distribution and highlights any potential outliers.

### ARIMA Model
1. Grid Search for ARIMA Parameters

To optimize the parameters (p, d, q) for ARIMA, we used Grid Search.

2. ARIMA Forecasting
Once the optimal parameters are identified, we train the ARIMA model on the Trading Volume and make predictions:
```python
forecast_arima = arima_model.predict(n_periods=len(test_data))

# Rescale the predictions back to original values
forecast_arima_rescaled = scaler.inverse_transform(forecast_arima.reshape(-1, 1))
```

### LSTM Model
1. Building and Training the LSTM Model
We build an LSTM model to capture the sequential dependencies in stock data:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(LSTM(50))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, Y_train, epochs=10, batch_size=32)
```

2. LSTM Forecasting
We predict the test data using the trained LSTM model:

```python
lstm_predictions = lstm_model.predict(X_test)
lstm_predictions_rescaled = scaler.inverse_transform(lstm_predictions)
```

### Model Evaluation and Comparison
To evaluate and compare the performance of ARIMA and LSTM, we use two evaluation metrics:

Mean Squared Error (MSE)
Mean Absolute Error (MAE)

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ARIMA evaluation
mse_arima = mean_squared_error(trading_volume[split:], forecast_arima_rescaled)
mae_arima = mean_absolute_error(trading_volume[split:], forecast_arima_rescaled)

# LSTM evaluation
mse_lstm = mean_squared_error(trading_volume[split+time_steps:], lstm_predictions_rescaled)
mae_lstm = mean_absolute_error(trading_volume[split+time_steps:], lstm_predictions_rescaled)

print(f'ARIMA - MSE: {mse_arima}, MAE: {mae_arima}')
print(f'LSTM - MSE: {mse_lstm}, MAE: {mae_lstm}')
```

Plotting Results
We can visualize the predicted vs actual Trading Volume for both models:
```python
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index[split:], trading_volume[split:], label='Actual Trading Volume', color='blue')
plt.plot(stock_data.index[split:], forecast_arima_rescaled, label='ARIMA Predictions', color='red')
plt.plot(stock_data.index[split+time_steps:], lstm_predictions_rescaled, label='LSTM Predictions', color='green')

plt.title('Trading Volume Predictions: ARIMA vs LSTM')
plt.xlabel('Date')
plt.ylabel('Trading Volume')
plt.legend()
plt.show()
```

## Conclusion

**ARIMA**: A traditional statistical model that works well for linear time series but may fail to capture complex patterns in stock data.

**LSTM**: A deep learning model that excels in capturing non-linear trends and long-term dependencies but requires more computational power and data.

### In our comparison:

- **LSTM** generally performs better for complex, non-linear stock price data, capturing patterns missed by ARIMA.
- **ARIMA** can be useful for simpler time series with less volatility.

This project demonstrates how two different approaches to time series forecasting can be implemented and evaluated on stock market data. Each model has its strengths and weaknesses, and the choice of the model depends on the nature of the data and the forecasting goals.
