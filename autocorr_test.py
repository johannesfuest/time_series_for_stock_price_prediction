import argparse
from ljung_box_pierce import lbp_test
import os
import pandas as pd
from tqdm import tqdm
from typing import List

def parse_args():
    """
    Parse command line arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description='Identify inefficient files')
    parser.add_argument('--output_file', type=str, default='autocorr_test.csv', help='Output file to save inefficient files to')
    return parser.parse_args()


def get_inefficient_files(files: List[str], output_file: str) -> None:
    """
    This function reads in a list of files and checks if the time series exhibit significant autocorrelation by 
    performing the Ljung-Box-Pierce test. All files that exhibit significant autocorrelation are written to the output
    file.
    
    :param files: A list of files to check for inefficiency.
    :param output_file: The name of the file to write inefficient files to.
    
    :return: None
    """
    with open(f'data/results/{output_file}', 'w') as f:
        f.write('file|p-value|autocorrelation\n')
        for file in tqdm(files, desc='Checking files for autocorrelation...'):
            data = pd.read_csv(f'data/cleaned/{file}')
            try:
                p_value, autocorrelated = lbp_test(data['close'].pct_change().dropna(), p=0, q=0)
            except ValueError:
                continue
            f.write(f'{file.split('_')[1]}|{p_value}|{autocorrelated}\n')
    return

if __name__=='__main__':
    os.chdir(os.path.dirname(__file__))
    all_files = os.listdir('./data/cleaned/')
    all_files = [f'{file}' for file in all_files if file.endswith('.csv')]
    inefficient_files = get_inefficient_files(all_files, parse_args().output_file)
    