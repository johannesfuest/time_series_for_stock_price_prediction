import numpy as np
import pandas as pd
import os
from scipy.stats import runs
import argparse

def runs_test_manual(data):
    """Manually compute the runs test for a given pandas Series."""
    # Convert to binary sequence
    median_val = data.median()
    binary_sequence = (data > median_val).astype(int)

    # Count runs
    shifts = np.diff(binary_sequence)
    num_runs = np.sum(shifts != 0) + 1

    # Calculate expected runs and variance
    n1 = np.sum(binary_sequence == 1)
    n0 = np.sum(binary_sequence == 0)
    N = len(data)
    
    expected_runs = 1 + 2 * n1 * n0 / N
    variance_runs = 2 * n1 * n0 * (2 * n1 * n0 - N) / (N**2 * (N - 1))

    # Z-statistic
    if variance_runs == 0:
        return None  # Avoid division by zero
    Z = (num_runs - expected_runs) / np.sqrt(variance_runs)

    return Z, num_runs, expected_runs

def process_file(filepath):
    """ Process an individual file to perform the runs test. """
    try:
        data = pd.read_csv(filepath)
        if 'Close' not in data.columns:
            raise ValueError(f"No 'Close' column in {filepath}")
        
        z_stat, num_runs, expected_runs = runs_test_manual(data['Close'])
        print(f"Runs Test for {os.path.basename(filepath)}: Z-statistic = {z_stat}, Runs = {num_runs}, Expected Runs = {expected_runs}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main(directory):
    """ Process all CSV files in the specified directory. """
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            process_file(filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform manual runs tests on stock price data.")
    parser.add_argument("directory", type=str, help="Directory containing CSV files")
    args = parser.parse_args()
    main(args.directory)
