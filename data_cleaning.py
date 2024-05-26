import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import List


def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument("--dir_name", type=str, default="data/raw/",
                        help="Name of the directory containing the csv files to be cleaned.")
    parser.add_argument("--file_prefix", type=str, default="cleaned_",
                        help="prefix added to file names saved after cleaning. Pass empty string to overwrite.")
    parser.add_argument("--dev", action="store_true", 
                        help="Set this flag to run processing using only first file in the directory for debugging.")
    args = parser.parse_args()
    return args

def clean_data(file_names: List[str], file_prefix: str, dir_name: str, dev: bool) -> None:
    """
    Cleans the data in the csv files by removing rows with missing values.
    Saves the cleaned files with the prefix added to the file names.
    Additionally, saves a diagnostics dataframe with the number of rows before and after cleaning, as well as some other
    metrics.
    
    :param file_names: List of file names to be cleaned.
    :param file_prefix: Prefix to be added to the file names saved after cleaning.
    :param dir_name: Name of the directory containing the csv files to be cleaned.
    :param dev: Set this flag to run processing using only first file in the directory for debugging.
    
    :return: None
    """
    
    diagnostics = pd.DataFrame(columns=[
        "file_name", "n_rows_before", "n_rows_after", "n_rows_removed", "n_missing_date", "n_missing_open",
        "n_missing_high", "n_missing_low", "n_missing_close", "percent_missing_date", "percent_missing_open",
        "percent_missing_high", "percent_missing_low", "percent_missing_close", "max_date_diff", "n_date_diffs_above_2"])
    
    for file_name in tqdm(file_names, desc="Cleaning files"):
        df = pd.read_csv(f"{dir_name}/{file_name}")
        n_rows_before = df.shape[0]
        
        # Ensure correct data types
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df["low"] = pd.to_numeric(df["low"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        
        # Compute diagnostics columns and append to diagnostics df
        n_missing_date = df["date"].isnull().sum()
        n_missing_open = df["open"].isnull().sum()
        n_missing_high = df["high"].isnull().sum()
        n_missing_low = df["low"].isnull().sum()
        n_missing_close = df["close"].isnull().sum()
        
        percent_missing_date = n_missing_date/n_rows_before
        percent_missing_open = n_missing_open/n_rows_before
        percent_missing_high = n_missing_high/n_rows_before
        percent_missing_low = n_missing_low/n_rows_before
        percent_missing_close = n_missing_close/n_rows_before
        
        
        df = df.dropna()
        df["date_diff"] = df["date"].diff()
        n_rows_after = df.shape[0]
        max_date_diff = df["date_diff"].max()
        n_date_diffs_above_2 = df[df["date_diff"] > pd.Timedelta(days=2)].shape[0]
        n_rows_removed = n_rows_before - n_rows_after
        start_date = df["date"].min()
        end_date = df["date"].max()
        
        diagnostics = pd.concat([diagnostics, pd.DataFrame({
            "file_name": [file_name], "n_rows_before": [n_rows_before], "n_rows_after": [n_rows_after],
            "n_rows_removed": [n_rows_removed], "n_missing_date": [n_missing_date], "n_missing_open": [n_missing_open],
            "n_missing_high": [n_missing_high], "n_missing_low": [n_missing_low], "n_missing_close": [n_missing_close],
            "percent_missing_date": [percent_missing_date], "percent_missing_open": [percent_missing_open],
            "percent_missing_high": [percent_missing_high], "percent_missing_low": [percent_missing_low],
            "percent_missing_close": [percent_missing_close], "start_date": [start_date], "end_date": [end_date],
            "max_date_diff": [max_date_diff], "n_date_diffs_above_2": [n_date_diffs_above_2]
        })])
        
        # Save cleaned file
        df = df.reset_index(drop=True)
        df.to_csv(f"data/cleaned/{file_prefix+file_name}", index=False)
        if dev:
            break
    # Save diagnostics file
    diagnostics.to_csv("diagnostics.csv", index=False)
    return None

if __name__ == "__main__":
    args = parse_arguments()
    all_files = os.listdir(args.dir_name)
    file_names = [file for file in all_files if file.endswith(".csv")]
    clean_data(file_names, args.file_prefix, args.dir_name, args.dev)
        

