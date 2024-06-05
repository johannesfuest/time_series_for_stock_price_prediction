import numpy as np
import pandas as pd
import os
from arch.unitroot import ADF
import argparse
from tqdm import tqdm

def run_unit_root_test(data):
    """
    Runs the Unit Root Test on the passed Pandas Series. Returns test stat & p-value
    """
    adf = ADF(data.copy(), trend='c', max_lags=20)
    return adf.stat, adf.pvalue


def test_data(directory):
    """
    Applies Unit Root Test to all Time Series CSVs in specified directory.
    Returns: Dataframe of Stationary Time Series Information:
        FilePath | Test Stat | Pvalue
    """
    data_list = []
    for filename in tqdm(os.listdir(directory), desc='Processing files...'):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            try:
                # Load data
                data = pd.read_csv(filepath)
                if 'close' not in data.columns:
                    raise ValueError(f"No 'close' column in {filepath}")        
                # Perform the unit root test
                if len(data['close']) < 60:
                    raise ValueError("Less than 60 Stock Observations")
                test_stat, p_value = run_unit_root_test(data['close'])
                test_stat_price, p_value_price = run_unit_root_test(data['close'])
                data_list.append([filename,test_stat_price, p_value_price, p_value_price > 0.05, test_stat,p_value, p_value > 0.05])
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    return pd.DataFrame(data_list, columns = ['file', 'test_stat_price', 'p_val_price', 'price_unit_root', 'test_stat_ret', 'p_val_ret', 'ret_unit_root'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform unit root tests on stock price data.")
    parser.add_argument("--directory", type=str, help="Directory containing CSV files", 
                        default="data/cleaned")
    parser.add_argument("--output_file", type=str, default='unit_root_test.csv',
                        help="Output file to save test results")
    args = parser.parse_args()
    stationary_ts_df = test_data(args.directory)
    stationary_ts_df.to_csv(f'data/results/{args.output_file}', index=False)