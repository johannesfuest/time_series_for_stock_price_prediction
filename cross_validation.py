import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def sarima_cv(file_path, order, seasonal_order, split_num):
    """
    Perform time series cross-validation on SARIMA model. 

    Given n splits, we run the following iteration:

        for n=1 to split_num:
            train on (n-1) splits, predict on nth split & record RMSE
    
    Parameters:
        file_path : explicit path to the time series data.
        order (tuple): The (p, d, q) order of the model for the number of AR parameters, differences, and MA parameters.
        seasonal_order (tuple): The (P, D, Q, s) seasonal order of the model.
        split_num (int): Number of splits for cross-validation.
    
    Returns:
        list: RMSE scores for each split.
    """

    #read in the data
    data_df = pd.read_csv(file_path)
    data_df['date'] = pd.to_datetime(data_df['date'])
    data_df.set_index('date', inplace=True)
    data = data_df['close']

    n_records = len(data)
    #print(f"{n_records} total samples")
    if n_records < split_num:
        raise ValueError("Number of splits is greater than the number of data points.")
    
    test_size = n_records // split_num
    start_train_size = n_records - (test_size * split_num)
    
    rmse_scores = []

    for i in range(split_num):

        train_size = n_records - (split_num - i) * test_size

        #print(f"Train up to split {i}, test on split {i+1}")
        #print(f"Train on 0 to {train_size}, test on split {train_size} to {train_size + test_size}")
        
        train, test = data.iloc[:train_size], data.iloc[train_size:train_size + test_size]

        model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=0)
        predictions = model_fit.forecast(len(test))
        
        rmse = np.sqrt(mean_squared_error(test, predictions))
        rmse_scores.append(rmse)
    
    return rmse_scores