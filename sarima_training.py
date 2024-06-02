import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
import itertools
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

def sarima_training(file_path, p_values, d_values, q_values, P_values, D_values, Q_values, s_values, split_num):
    data = pd.read_csv(file_path)    
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    # Selecting the 'close' price for modeling
    close_prices = data['close']
    # Splitting the dataset into training and testing sets
    train_size = int(len(close_prices) * 0.8)
    train, test = close_prices.iloc[:train_size], close_prices.iloc[train_size:]
    best_rmse = np.inf
    best_cfg = None
    
    # Grid search
    for p, d, q, P, D, Q, s in itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values, s_values):
        order = (p, d, q)
        seasonal_order = (P, D, Q, s)
        try:
            rmse_scores = sarima_cv(data, order, seasonal_order, split_num)
            # take average of rmse scores
            avg = statistics.mean(rmse_scores)
            if avg < best_rmse:
                best_rmse, best_cfg = avg, (order, seasonal_order)
            print(f'Tested: {order}x{seasonal_order} with RMSE={avg}')
        except:
            continue

    print(f'Best SARIMA: {best_cfg[0]}x{best_cfg[1]} with RMSE={best_rmse}')
    return best_cfg


