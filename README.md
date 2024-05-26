# Time Series for Stock Price Prediction

In this project we built an automated time series forecasting framework for stock prediction and tested it on over 3000 Nasdaq listed stocks. 

## Folder Structure

- **data**: This folder contains the project's data in CSV format.
- **dev**: This folder contains work in progress
- **pipeline.py**: This file contains our automated time series forecasting training pipeline
- **evaluation.py**: This file contains a script that measures the performance of our models.

## Installation

To ensure a clean and isolated environment, we recommed using a virtual environment. Here's how you can set up the environment:

1. **Create a Virtual Environment**: Run the following command in your terminal to create a virtual environment named `venv`:

    ```
    python -m venv venv
    ```

2. **Activate the Virtual Environment**: Activate the virtual environment using the appropriate command for your operating system:

    For macOS/Linux:

    ```
    source venv/bin/activate
    ```

    For Windows:

    ```
    venv\Scripts\activate
    ```

3. **Install Requirements**: Once the virtual environment is activated, install the project dependencies using pip:

    ```
    pip install -r requirements.txt
    ```

4. **Download Data: Populate the data dictionary by downloading data from kaggle:

    Head to [kaggle] (https://www.kaggle.com/datasets/svaningelgem/nasdaq-daily-stock-prices/data) and download and unzip the data and move it to the data/raw directory:
    ```
    unzip downloads/archive.zip
    mv downloads/archive/*.csv time_series_for_stock_price_prediction/data/raw
    ```
