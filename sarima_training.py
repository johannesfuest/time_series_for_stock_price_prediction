import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
import itertools

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


