import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima

def sarima_training(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    
    # Selecting the 'close' price for modeling
    close_prices = data['close']
    
    # Splitting the dataset into training and testing sets
    split_point = len(close_prices) // 2
    train, test = close_prices.iloc[:split_point], close_prices.iloc[split_point:]
    
    # Fit the SARIMA model
    model = auto_arima(train, seasonal=True, m=12,
                       trace=True, error_action='ignore', suppress_warnings=True)
    
    sarima_model = SARIMAX(train, order=model.order, seasonal_order=model.seasonal_order,
                           enforce_stationarity=False, enforce_invertibility=False)
    sarima_model_fit = sarima_model.fit(disp=0)
    
    # Make predictions
    predictions = sarima_model_fit.forecast(len(test))
    
    # Calculate MSE
    mse = mean_squared_error(test, predictions)
    return mse

