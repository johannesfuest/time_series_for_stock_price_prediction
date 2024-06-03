import pandas as pd
from sarima_training import sarima_training
import os
import argparse
from tqdm import tqdm
from typing import List

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description='Find the best model for time series data')
    parser.add_argument('--output_file', type=str, help='Output file to save best models configs to')
    parser.add_argument('--sample', action='store_true', help='Run on subsample of 100 stocks')
    parser.add_argument('--rolling', action='store_true', help='Refit model on rolling window when predicting')
    parser.add_argument('--split_num', type=int, default=4, help='Number of splits for cross-validation')
    parser.add_argument('--predict', action='store_true', help='Predict on test data using best model configs')
    return parser.parse_args()


def get_best_configs(output_file: str, df_autocorrs: pd.DataFrame, p_values: List[int] = list(range(0, 4)), 
                     d_values: List[int] = list(range(0, 3)), q_values: List[int] = list(range(0, 4)),  
                     split_num: int = 4, sample: bool = False, predict: bool = False, rolling: bool = False) -> None:
    """
    This function finds the best SARIMA model configurations for a list of time series data files by performing a grid 
    search and cross-validation. The best configurations are written to the output file.
    
    :param output_file: The name of the file to write best model configurations to.
    :param df_autocorrs: A DataFrame containing the results of the Ljung-Box-Pierce test for each time series.
    :param p_values: A tuple of the range of AR parameters to search.
    :param d_values: A tuple of the range of differencing parameters to search.
    :param q_values: A tuple of the range of MA parameters to search.
    :param split_num: Number of splits for cross-validation.
    :param sample: Whether to run on a sample of 100 stocks.
    :param predict: Whether to predict on test data using best model configurations.
    :param rolling: Whether to refit model on rolling basis when predicting.
    
    :return: None
    """
    if sample:
        df_diagnostics = pd.read_csv('diagnostics.csv')
        df_autocorrs['file_name'] = df_autocorrs['file_name'].apply(lambda x: x.split('_')[1])
        df_autocorrs = pd.merge(df_autocorrs, df_diagnostics, on='file_name', how='left')
        df_autocorrs = df_autocorrs.query('n_rows_after >= 1000 and n_rows_after <= 2000').copy()
        df_autocorrs['file_name'] = df_autocorrs['file_name'].apply(lambda x: f'cleaned_{x}')
        df_autocorrs = df_autocorrs.sample(100)
        
    with open(output_file, 'a') as f:
        f.write('file|best_cfg|avg_rmse_across_folds|p_val|resids_autocorrelated\n')
    for file in tqdm(df_autocorrs['file_name'], desc='Finding best model configurations...'):
        # try:
        best_cfg, best_rmse, p_val, autocorrelated = sarima_training(f'data/cleaned/{file}', p_values, d_values, 
                                                                     q_values, split_num, predict, rolling)
        with open(output_file, 'a') as f1:
            f1.write(f'{file}|{best_cfg}|{best_rmse}|{p_val}|{autocorrelated}\n')
        # except:
        #     with open(output_file, 'a') as f1:
        #         f1.write(f'{file}|{'error'}|{'error'}|{'error'}|{'error'}\n')
        #     continue

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    df_autocorrs = pd.read_csv('autocorr_test.csv', sep='|')
    df_autocorrs = df_autocorrs[df_autocorrs['autocorrelation'] == True]
    get_best_configs(parse_args().output_file, df_autocorrs, sample=parse_args().sample)
    
    