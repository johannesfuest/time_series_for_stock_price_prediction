# ARIMA for Price Prediction of Inefficient Stocks


<details>
  <summary>Summary</summary>
In this short project we applied weak-form market effiency tests to over 3000 Nasdaq listed stocks. For each stock we had daily opening, closing, 
high, and low prices over the past 50 years (depending on the company's status and when it was listed of course).

The four tests we applied were:
    1. Autocorrelation test
    2. Runs test
    3. Augmented Dickey-Fuller test
    4. Variance Ratio test

We saw great heterogeneity in efficiency among stocks, as well as among the tests' abilities to detect inefficiencies (e.g the unit root test deemed 
only around 22% of stocks inefficient, whereas this number was 86% for the variance ratio test). This demonstrates the different ways in which the 
time-series demonstrate non-randomness. After running efficiency tests that tested the null hypothesis that the data were generated by a random walk 
against against various alternatives,  we were then interested in whether arima models could be automically fit to "inefficient" stocks using a grid 
search with cross validation approach.

Due to computational constraints we limited our analysis to 100 randomly sampled stocks.
Generally, the arima models performed rather poorly (see the analysis notebook). We believe key reasons for this were:
    1. Lack of seasonality in our models (this was also a computational limitation)
    2. Complex shapes of the time series' pacfs and acfs

Overall, the results were close to expectation. Potential extensions would include:
    1. Introducing seasonality by moving to SARIMA models
    2. Increasing the sample size.
</details>


<details>
  <summary>Project Structure</summary>
    
- **data**: This folder contains the project's data in CSV format.
- **dev**: This folder contains work in progress
- **pipeline.py**: This file contains our automated time series forecasting training pipeline
- **evaluation.py**: This file contains a script that measures the performance of our models.
</details>


<details>
  <summary>Installation</summary>
We recommed using a virtual environment. Here's how you can set up the environment:

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

4. **Download Data**: Populate the data dictionary by downloading data from kaggle:

    Head to [kaggle](https://www.kaggle.com/datasets/svaningelgem/nasdaq-daily-stock-prices/data) and download and unzip the Nasdaq stock price dataset and move it to the data/raw directory:
</details>

<details>
  <summary>Running Experiments</summary>
To run the experiments and replicate the results of our projects, run the following commands:

1. **Data Preprocessing**: This will created cleaned versions of all time series stored in the cleaned dir.

    ```
    python data_cleaning.py
    ```
2. **Run Weak-Form Tests**: This runs the autocorrelation, sample, unit root and variance ratio tests on all 3k stocks and saves the respective results to dataframes in the results dir.

    ```
    python autocorr_test.py
    python runs_test_package.py
    python unit_root_test.py
    python variance_ratio_test.py
    ```
3. **Train sample of 100 SARIMA models** This trains 100 ARIMA models using grid search and cross validation, and generates predictions based on them for the final 30 trading days in the data, which are saved to the predictions dir.
    ```
    python best_model_search.py --output_file sample_100_n.csv --sample --rolling --predict --n_test 30
    ```
4. **Analyse Results** Run the results_analysis.ipynb to generate the plots and insights discussed in the project paper. 
</details>
