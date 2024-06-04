import numpy as np
import pandas as pd
import os
from statsmodels.sandbox.stats.runs import runstest_1samp
import argparse
from tqdm import tqdm

def runs_test_package(data):
    """ Perform the runs test on an inputted pandas Series column and return test statistic / p-value. """
    # Convert data to binary format (1 if value is above median, 0 otherwise)
    median_val = data.median()
    binary_sequence = (data > median_val).astype(int)    
    # Perform the runs test
    test_stat, p_value = runstest_1samp(binary_sequence, correction=False)
    return test_stat, p_value

def process_file(filepath, output_file=None):
    """ file processing for the test """
    try:
        # Load data
        data = pd.read_csv(filepath)
        if 'close' not in data.columns:
            raise ValueError(f"No 'Close' column in {filepath}")        
        # Perform the runs test
        test_stat, p_value = runs_test_package(data['close'])
        if p_value < 0.05:
            random = False
        else:
            random = True
        # Write results to output file
        if output_file:
            with open(f'data/results/{output_file}', 'a') as f:
                f.write(f"{filepath.split('_')[1]}|{test_stat}|{p_value}|{random}\n")
    except Exception as e:
        with open(f'data/results/{output_file}', 'a') as f:
            f.write(f"{filepath.split('_')[1]}|error|error|error\n")
        print(f"Error processing {filepath}: {e}")

def main(directory, output_file=None):
    """ Process all CSV files in the specified directory. """
    with open(f'data/results/{output_file}', 'w') as f:
        f.write('file|test_statistic|p_value|random\n')
    for filename in tqdm(os.listdir(directory), desc='Processing files...'):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            process_file(filepath, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform runs tests on stock price data.")
    parser.add_argument("--directory", type=str, default='data/cleaned', help="Directory containing CSV files")
    parser.add_argument("--output_file", type=str, default='runs_test.csv',help="Output file to save test results")
    args = parser.parse_args()
    main(args.directory, args.output_file)
