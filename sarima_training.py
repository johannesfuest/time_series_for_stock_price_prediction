from ljung_box_pierce import lbp_test
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
import itertools
import statistics
from tqdm import tqdm
import warnings


# Suppress warnings related to ARIMA convergence
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def sarima_cv(file_path, order, split_num):
    """
    Perform time series cross-validation on SARIMA model. 
    Given n splits, we run the following iteration:
        for n=1 to split_num:
            train on (n-1) splits, predict on nth split & record RMSE

    Parameters:
        file_path : explicit path to the time series data.
        order (tuple): The (p, d, q) order of the model for the number of AR parameters, differences, and MA parameters.
        split_num (int): Number of splits for cross-validation.
    
    Returns:
        list: RMSE scores for each split.
    """
    TRAINING_PERCENT = 0.9
    data_df = pd.read_csv(file_path)
    data = data_df['close']
    data_train = data[:int(TRAINING_PERCENT*len(data))]
    
    n_records = len(data_train)
    
    if n_records < split_num:
        raise ValueError("Number of splits is greater than the number of data points.")
    
    cutoffs = np.linspace(int(0.7*n_records), n_records, split_num + 1, dtype=int)
    rmse_scores = []
    for i in range(0, split_num):
        train, test = data.iloc[:cutoffs[i]], data.iloc[cutoffs[i]:cutoffs[i+1]]
        model = ARIMA(train, order=order)
        model_fit = model.fit()
        predictions = model_fit.forecast(len(test))
        rmse = np.sqrt(mean_squared_error(test, predictions))
        rmse_scores.append(rmse)
    return rmse_scores


def get_model_predictions(file_path, order, rolling=False):
    TRAINING_PERCENT = 0.95
    data_df = pd.read_csv(file_path)
    data = data_df['close']
    resids = []
    preds = []
    actuals = []
    if rolling:
        for i in range(int(TRAINING_PERCENT*len(data)), len(data)-2):
            data_train_temp = data[:i]
            data_test_temp = data[i:]
            model = ARIMA(data_train_temp, order=order)
            model_fit = model.fit()
            predictions = model_fit.forecast(1)
            resids.append(data_test_temp.iloc[0] - predictions.iloc[0])
            preds.append(predictions.iloc[0])
            actuals.append(data_test_temp.iloc[0])
    else:
        model = ARIMA(data[:int(TRAINING_PERCENT*len(data))], order=order)
        model_fit = model.fit()
        predictions = model_fit.forecast(len(data) - int(TRAINING_PERCENT*len(data)))
        resids = data[int(TRAINING_PERCENT*len(data)):] - predictions
        preds = predictions
        actuals = data[int(TRAINING_PERCENT*len(data)):]  
    ticker_name = file_path.split("/")[-1].split(".")[0]
    if not rolling:
        with open(f"./data/predictions/{ticker_name}.csv", "a") as f:
            f.write('pred|actual\n')
            for pred, act in zip(preds, actuals):
                f.write(f'{pred}|{act}\n')
    else:
        with open(f"./data/predictions/{ticker_name}_rolling.csv", "a") as f:
            f.write('pred|actual\n')
            for pred, act in zip(preds, actuals):
                f.write(f'{pred}|{act}\n')
                
    return


def test_model_significance(file_path, order):
    TRAINING_PERCENT = 0.9
    data_df = pd.read_csv(file_path)
    data = data_df['close']
    data = data[:int(TRAINING_PERCENT*len(data))]
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    residuals = model_fit.resid
    p_value, autocorrelated = lbp_test(residuals, order[0], order[2], k=min(20, len(residuals) - 1), significance=0.05)
    return p_value, autocorrelated
    

def sarima_training(file_path, p_values, d_values, q_values, split_num, predict= False, rolling=False):
    best_rmse = np.inf
    best_cfg = None
    
    # Grid search
    for p, d, q in itertools.product(p_values, d_values, q_values):
        order = (p, d, q)
        rmse_scores = sarima_cv(file_path, order, split_num)
        # take average of rmse scores
        avg = statistics.mean(rmse_scores)
        if avg < best_rmse:
            best_rmse, best_cfg = avg, (order)
    # Check significance of best model
    p_value, autocorrelated = test_model_significance(file_path, best_cfg)
    if predict:
        get_model_predictions(file_path, best_cfg, rolling)
    return best_cfg, best_rmse, p_value, autocorrelated
