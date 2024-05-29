
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import List

np.random.seed(0)

def lbp_test(resids: List[float], p:int, q:int, k: int=20, significance: float=0.05) -> bool:
    """
    This function performs the Ljung-Box-Pierce test on the residuals of a time series model.
    The null hypothesis is that the residuals are independently distributed.
    If the p-value is less than the significance level, the null hypothesis is rejected.
    
    :param resids: List of residuals from the time series model.
    :param p: Number of AR terms in the model.
    :param q: Number of MA terms in the model.
    :param k: Number of lags to include in the test. 20 by default based on lecture notes page 48.
    :param significance: Significance level for the test. 0.05 by default.
    
    :return: True if the null hypothesis is rejected, False otherwise.
    """
    #TODO: check degrees of freedom for ARMA variants
    #TODO: check seasonal argument period for seasonal ARIMA
    lb_test = sm.stats.acorr_ljungbox(resids, model_df = p+q, lags=k, return_df=True)
    return lb_test.iloc[(k-1), 1] < significance

if __name__ == "__main__":
    # Test the function with random time series data against ARMA(1,1) model
    random_ts = np.random.normal(0, 1, 100)
    lbp_test(random_ts, 1, 1)
    # Test rejection of null hypothesis at 5% significance level as expected.