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
    return parser.parse_args()


def get_best_configs(output_file: str, df_autocorrs: pd.DataFrame, p_values: List[int] = list(range(0, 5)), 
                     d_values: List[int] = list(range(0, 2)), q_values: List[int] = list(range(0, 5)),  
                     split_num: int = 5) -> None:
    """
    This function finds the best SARIMA model configurations for a list of time series data files by performing a grid 
    search and cross-validation. The best configurations are written to the output file.
    
    :param output_file: The name of the file to write best model configurations to.
    :param df_autocorrs: A DataFrame containing the results of the Ljung-Box-Pierce test for each time series.
    :param p_values: A tuple of the range of AR parameters to search.
    :param d_values: A tuple of the range of differencing parameters to search.
    :param q_values: A tuple of the range of MA parameters to search.
    :param split_num: Number of splits for cross-validation.
    
    :return: None
    """
    with open(output_file, 'w') as f:
        f.write('file|best_cfg|avg_rmse_across_folds|p_val|resids_autocorrelated\n')
    for file in tqdm(df_autocorrs['file'], desc='Finding best model configurations...'):
        best_cfg, best_rmse, p_val, autocorrelated = sarima_training(f'data/cleaned/{file}', p_values, d_values, q_values, split_num)
        with open(output_file, 'w') as f:
            f.write(f'{file}|{best_cfg}|{best_rmse}|{p_val}|{autocorrelated}\n')

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    df_autocorrs = pd.read_csv('autocorr_test.csv', sep='|')
    df_autocorrs = df_autocorrs[df_autocorrs['autocorrelation'] == True]
    get_best_configs(parse_args().output_file, df_autocorrs)
    