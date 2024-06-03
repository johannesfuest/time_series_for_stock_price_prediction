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
    adf = ADF(data.copy())
    return adf.stat, adf.pvalue


def test_data(directory):
    """
    Applies Unit Root Test to all Time Series CSVs in specified directory.
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
                # Perform the unit root test
                if len(data['close']) < 60:
                    raise ValueError("Less than 60 Stock Observations")
                test_stat, p_value = run_unit_root_test(data['close'])
                if p_value<=0.05:
                    data_list.append([filename,test_stat,p_value])
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    return pd.DataFrame(data_list, columns = ['Filename', 'Test Stat', 'p-value'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform unit root tests on stock price data.")
    parser.add_argument("--directory", type=str, help="Directory containing CSV files", 
                        default="/Users/jonathan.williams/Desktop/Stats_207_Final_Project/time_series_for_stock_price_prediction/data/cleaned")
    args = parser.parse_args()
    stationary_ts_df = test_data(args.directory)
    print(stationary_ts_df)
    stationary_ts_df.to_csv("passed_unit_root_ts.csv", index=False)