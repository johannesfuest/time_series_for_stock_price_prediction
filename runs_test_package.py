import numpy as np
import pandas as pd
import os
from scipy.stats import runs
import argparse

def runs_test_package(data):
    """ Perform the runs test on an inputted pandas Series column and return test statistic / p-value. """
    # Convert data to binary format (1 if value is above median, 0 otherwise)
    median_val = data.median()
    binary_sequence = (data > median_val).astype(int)    
    # Perform the runs test
    test_stat, p_value = runs(binary_sequence, correction=True)
    return test_stat, p_value

def process_file(filepath):
    """ file processing for the test """
    try:
        # Load data
        data = pd.read_csv(filepath)
        if 'Close' not in data.columns:
            raise ValueError(f"No 'Close' column in {filepath}")        
        # Perform the runs test
        test_stat, p_value = runs_test_package(data['Close'])
        print(f"Runs Test for {os.path.basename(filepath)}: Test Statistic = {test_stat}, P-Value = {p_value}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main(directory):
    """ Process all CSV files in the specified directory. """
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            process_file(filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform runs tests on stock price data.")
    parser.add_argument("directory", type=str, help="Directory containing CSV files")
    args = parser.parse_args()
    main(args.directory)
