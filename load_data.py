import pandas as pd

def load_ticker_data(ticker):
    """
    Load data for a given ticker symbol from CSV files located in the data/raw directory.

    Parameters:
    ticker (str): Ticker symbol of the stock.

    Returns:
    DataFrame: Pandas DataFrame containing the stock data.
    """
    file_path = f'data/raw/{ticker}.csv' 
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"No data found for ticker {ticker}. Please check the ticker symbol and data directory.")
        return None


