import numpy as np
import pandas as pd
import os
from arch.unitroot import VarianceRatio
import argparse
from tqdm import tqdm

def run_var_ratio_test(data):
    """
    Runs the Variance Ratio Test on the passed Pandas Series. Returns test stat & p-value
    """
    vr = VarianceRatio(data.copy(), lags=2)
    return vr.stat, vr.pvalue


def test_data(directory):
    """
    Applies Variance Ratio Test to all Time Series CSVs in specified directory.
    Returns: Dataframe of Stationary Time Series Information:
        FilePath | Test Stat | Pvalue
    """
    data_list = []
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            try:
                # Load data
                data = pd.read_csv(filepath)
                if 'close' not in data.columns:
                    raise ValueError(f"No 'close' column in {filepath}")        
                # Perform the var ratio test
                if len(data['close']) < 60:
                    raise ValueError("Less than 60 Stock Observations")
                test_stat, p_value = run_var_ratio_test(data['close'].pct_change().dropna())
                data_list.append([filename,test_stat,p_value, p_value > 0.05])
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    return pd.DataFrame(data_list, columns = ['file', 'test_stat', 'p_val', 'data_from_random_walk'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform variance ratio tests on stock price data.")
    parser.add_argument("--directory", type=str, help="Directory containing CSV files", 
                        default="data/cleaned")
    parser.add_argument("--output_file", type=str, default='var_ratio_test.csv',
                        help="Output file to save test results")
    args = parser.parse_args()
    correlated_ts_df = test_data(args.directory)
    print(f'Number of random walk time series: {correlated_ts_df["data_from_random_walk"].sum()}')
    correlated_ts_df.to_csv(f"data/results/{args.output_file}", index=False)