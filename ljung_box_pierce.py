
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import List

np.random.seed(0)

def lbp_test(series: List[float], p:int, q:int, k: int=20, significance: float=0.05) -> float:
    """
    This function performs the Ljung-Box-Pierce test on a time series
    The null hypothesis is that the time series is independently distributed.
    If the p-value is less than the significance level, the null hypothesis is rejected.
    Returns the p-value of the test, as well as whether the null hypothesis is rejected.
    
    :param series: A time series that is to be tested for significant autocorrelation. Could be residuals from a model.
    :param p: Number of AR terms in the model.
    :param q: Number of MA terms in the model.
    :param k: Number of lags to include in the test. 20 by default based on lecture notes page 48.
    :param significance: Significance level for the test. 0.05 by default.
    
    :return: The p-value of the test.
    """
    if len(series) <= k:
        raise ValueError("Time series must be longer than the number of lags.")
    
    lb_test = sm.stats.acorr_ljungbox(series, model_df = p+q, lags=k, return_df=True)
    return lb_test.iloc[(k-1), 1], lb_test.iloc[(k-1), 1] < significance

if __name__ == "__main__":
    # Test the function with random time series data against ARMA(1,1) model
    random_ts = np.random.normal(0, 1, 100)
    print(lbp_test(random_ts, 0, 0))
    # Test correctly detecs time series as independent