"""
This module contains the ingestion logic for Financial Times Series (FTS) data. It provides functions to read, process, and store FTS data from various sources into a structured format suitable for analysis and modeling.
"""
import pandas as pd

import yfinance as yf

from fts_tool.config import DEFAULT_DATA_DIR, PRICES_DATA_SOURCE

class YahooFinanceIngestor:
   """
   A class to handle the ingestion of financial time series data from Yahoo Finance.

   Attributes:
       ticker (str): The stock ticker symbol for which to fetch data.
       start_date (str): The start date for the data retrieval in 'YYYY-MM-DD' format.
       end_date (str): The end date for the data retrieval in 'YYYY-MM-DD' format.
   """

   def __init__(self, ticker: str, start_date: str, end_date: str, interval: str = "1d"):
       """
       Initializes the YahooFinanceIngestor with the specified ticker and date range.

       Args:
           ticker (str): The stock ticker symbol.
           start_date (str): The start date for data retrieval.
           end_date (str): The end date for data retrieval.
           interval (str): The interval for the time series data.
       """
       self.ticker = ticker
       self.start_date = start_date
       self.end_date = end_date
       self.interval = interval
       self.data_storage_path = PRICES_DATA_SOURCE / f"{self.ticker}" / f"{self.ticker}_{self.start_date}_{self.end_date}.csv"

   def fetch_data(self):
      """
      Fetches financial time series data from Yahoo Finance for the specified ticker and date range.

      Returns:
         pd.DataFrame: A DataFrame containing the fetched financial time series data.
      """
      # Implementation to fetch data from Yahoo Finance (locally or via API)

      # Check if the data exists partially in the local storage and fetch missing data
      if self.data_storage_path.parent.exists() and len(list(self.data_storage_path.parent.glob(f"{self.ticker}_*.csv"))) > 0:
         already_fetched_files = list(self.data_storage_path.parent.glob(f"{self.ticker}_*.csv"))

         data = pd.DataFrame()

         for file in already_fetched_files :
            # Load the existing data
            existing_data = pd.read_csv(file, index_col=0, parse_dates=True)
            existing_data.index = pd.to_datetime(existing_data.index)

            data = pd.concat([data, existing_data])

         data = data.sort_index().drop_duplicates()

         # Check if the requested date range is already covered by the existing data
         if self.start_date >= data.index.min().strftime('%Y-%m-%d') and self.end_date <= data.index.max().strftime('%Y-%m-%d'):
            # The requested date range is already covered by the existing data
            return data.loc[(data.index >= self.start_date) & (data.index <= self.end_date)]
         else:
            # Fetch just the missing data from Yahoo Finance
            missing_start_date = min(self.start_date, data.index.min().strftime('%Y-%m-%d'))
            missing_end_date = max(self.end_date, data.index.max().strftime('%Y-%m-%d'))

            fetched_data = yf.download(self.ticker, start=missing_start_date, end=missing_end_date, interval=self.interval)

            if isinstance(fetched_data.columns, pd.MultiIndex):
                     fetched_data.columns = [col[0] if col[0] != self.ticker else col[1] for col in fetched_data.columns.values]
            
            data = pd.concat([data, fetched_data])

            data = data.sort_index().drop_duplicates()

            # return existing date range if it is already covered by the existing data
            # paying attention to the fact that the start_date and end_date may be non working days, so we need to find the closest available dates in the data
            return data.loc[(data.index >= self.start_date) & (data.index <= self.end_date)]
      else:
         # Fetch the entire requested date range from Yahoo Finance
         data = yf.download(self.ticker, start=self.start_date, end=self.end_date, interval=self.interval)
         return data

   def process_data(self, data):
       """
       Processes the fetched financial time series data.

       Args:
           data (pd.DataFrame): The raw financial time series data.

       Returns:
           pd.DataFrame: A processed DataFrame ready for analysis or modeling.
       """
       # Implementation to process the data goes here
       
       data = data.dropna()

       return data

   def store_data(self, processed_data):
      """
      Stores the processed financial time series data into a structured format.

      Args:
         processed_data (pd.DataFrame): The processed financial time series data.
       
      Returns:
          None
      """
      # Implementation to store the processed data goes here

      if not self.data_storage_path.parent.exists():
         self.data_storage_path.parent.mkdir(parents=True, exist_ok=True)

      # Columns are multi-indexed, so we need to flatten them before saving
      if isinstance(processed_data.columns, pd.MultiIndex):
         processed_data.columns = [col[0] if col[0] != self.ticker else col[1] for col in processed_data.columns.values]

      processed_data.to_csv(self.data_storage_path)