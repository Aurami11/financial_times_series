"""
This module contains tools for analyzing Financial Time Series (FTS) data. It provides functions to compute various characteristics and metrics of FTS data, such as returns, log returns, volatitlity,
means, excess kurtosis, skewness, and other statistical properties. These tools are designed to facilitate the analysis and understanding of financial time series data for research and modeling purposes.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class FTSAnalyzer:
    """
    A class to analyze Financial Time Series (FTS) data and compute various characteristics and metrics.

    Attributes:
        data (pd.DataFrame): The financial time series data to be analyzed.
    """

    def __init__(self, data):
        """
        Initializes the FTSAnalyzer with the specified financial time series data.

        Args:
            data (pd.DataFrame): The financial time series data to be analyzed.
        """
        self.data = data
        self.returns = self.compute_returns()
        self.log_returns = self.compute_log_returns()

    def compute_returns(self):
        """
        Computes the returns of the financial time series data.

        Returns:
            pd.Series: A Series containing the computed returns.
        """
        return self.data.pct_change().dropna()

    def compute_log_returns(self):
        """
        Computes the log returns of the financial time series data.

        Returns:
            pd.Series: A Series containing the computed log returns.
        """

        returns = self.compute_returns()

        return returns.apply(lambda x: np.log(1 + x)).dropna()

    def define_ts(self, ts):
        """
        Defines the time series data to be analyzed.

        Args:
            ts (pd.Series): The time series data to be analyzed.
        """
        self.ts = ts

    def define_rolling_window(self, window):
        """
        Defines the rolling window size for computing rolling statistics.

        Args:
            window (int): The window size for calculating rolling statistics.
        """
        self.rolling_window = window

    def compute_std(self):
        """
        Computes the standard deviation of the financial time series data.

        Args:
            window (int): The window size for calculating rolling volatility.

        Returns:
            pd.Series: A Series containing the computed rolling volatility.
        """
        return self.ts.std()

    def compute_mean(self):
        """
        Computes the mean of the financial time series data.

        Returns:
            float: The computed mean of the time series data.
        """
        return self.ts.mean()
    
    def compute_min(self):
        """
        Computes the min of the financial time series data.
        
        Returns:
            float: The computed min of the time series data.
        """

        return self.ts.min()

    def compute_max(self):
        """
        Computes the max of the financial time series data.

        Returns:
              float: The computed max of the time series data.
        """

        return self.ts.max()
    
    def compute_excess_kurtosis(self):
         """
         Computes the excess kurtosis of the financial time series data.
   
         Returns:
               float: The computed excess kurtosis of the time series data.
         """
         return self.ts.kurtosis() - 3
    
    def compute_skewness(self):
         """
         Computes the skewness of the financial time series data.
   
         Returns:
               float: The computed skewness of the time series data.
         """
         return self.ts.skew()

    def plot_ts(self):
        """
        Plots the financial time series data.

        Returns:
            None
        """

        plt.figure(figsize=(12, 6))

        sns.lineplot(data=self.ts)
        plt.title("Financial Time Series Data")
        plt.ylabel("Value")
        plt.xlabel("Date")
        plt.show()
        
    def prices_plot(self) :
        """
        Plot the prices times series data

        Returns:
            None
        """

        plt.figure(figsize=(12,6))

        sns.lineplot(data=self.data)
        plt.title("Prices Times Series Data")
        plt.ylabel("Value")
        plt.xlabel("Date")
        plt.show()

    def normal_adjustment(self):
      """
      Adjust a normal distribution to the Times Series
      """

      ts_mean = self.compute_mean()
      ts_std = self.compute_std()

      plt.figure(figsize=(12, 6))
      sns.histplot(data=self.ts, kde=True, stat='density', bins=30, color='C0', edgecolor='black', alpha=0.6)

      x = np.linspace(self.ts.min(), self.ts.max(), 200)
      # protect against zero or NaN std
      if ts_std is None or np.isnan(ts_std) or ts_std <= 0:
         plt.title("Normal fit not available (std is zero or NaN)")
      else:
         pdf = (1.0 / (ts_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - ts_mean) / ts_std) ** 2)
         plt.plot(x, pdf, color='C1', lw=2, label='Normal PDF (fit)')
         plt.axvline(ts_mean, color='k', linestyle='--', label=f"Mean = {ts_mean:.4f}")

      plt.title("Histogram and Normal Fit")
      plt.xlabel("Value")
      plt.ylabel("Density")
      plt.legend()
      plt.show()

    def print_summary(self):
        """
        Prints a summary of the financial time series data, including mean, volatility, excess kurtosis, and skewness.

        Returns:
            None
        """
        print("Summary of Financial Time Series Data:")
        print(f"Mean: {self.compute_mean()}")
        print(f"Std: {self.compute_std()}")
        print(f"Min: {self.compute_min()}")
        print(f"Max: {self.compute_max()}")
        print(f"Excess Kurtosis: {self.compute_excess_kurtosis()}")
        print(f"Skewness: {self.compute_skewness()}")