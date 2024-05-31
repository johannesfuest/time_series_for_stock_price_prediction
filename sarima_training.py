import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
import itertools

def sarima_training(data, p_values, d_values, q_values, P_values, D_values, Q_values, s_values):
    # Given the data from the load_data function, as well as vectors of  values to try
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
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=0)
            predictions = model_fit.forecast(len(test))
            rmse = np.sqrt(mean_squared_error(test, predictions))
            if rmse < best_rmse:
                best_rmse, best_cfg = rmse, (order, seasonal_order)
            print(f'Tested: {order}x{seasonal_order} with RMSE={rmse}')
        except:
            continue

    print(f'Best SARIMA: {best_cfg[0]}x{best_cfg[1]} with RMSE={best_rmse}')
    return best_cfg


