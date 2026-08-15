"""
This module contains tools for analyzing Financial Time Series (FTS) data. It provides functions to compute various characteristics and metrics of FTS data, such as returns, log returns, volatitlity,
means, excess kurtosis, skewness, and other statistical properties. These tools are designed to facilitate the analysis and understanding of financial time series data for research and modeling purposes.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

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


    def null_hypothesis(self):
        """
        Runs basic null-hypothesis tests on the selected time series.
        The approximation used here is not the one for large sample (that is used in Chapter 1 of Analysis of Financial Times Series by Ruey S. Tsay)
        Tests include:
            - H0: mean = 0
            - H0: skewness = 0
            - H0: excess kurtosis = 0
            - Jarque-Bera test for normality

        Returns:
            dict: A dictionary containing test statistics and p-values.
        """
        if not hasattr(self, "ts") or self.ts is None:
            raise ValueError("Define a time series with define_ts() before running the null hypothesis tests.")

        x = pd.to_numeric(self.ts, errors="coerce").dropna()
        if len(x) < 2:
            raise ValueError("At least two valid observations are required for hypothesis testing.")

        n = len(x)
        mean = x.mean()
        std = x.std(ddof=1)
        skew = x.skew()
        excess_kurtosis = x.kurtosis()

        results = {}

        if std == 0 or pd.isna(std):
            results["null_mean"] = {
                "hypothesis": "H0: mean = 0",
                "statistic": np.nan,
                "p_value": np.nan,
                "reject_null": False,
                "note": "Standard deviation is zero, so the t-test is undefined."
            }
        else:
            t_stat = mean / (std / np.sqrt(n))
            p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1) if stats is not None else np.nan
            results["null_mean"] = {
                "hypothesis": "H0: mean = 0",
                "statistic": t_stat,
                "p_value": p_value,
                "reject_null": bool(stats is not None and p_value < 0.05),
                "note": "Two-sided one-sample t-test."
            }

        skew_se = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
        skew_z = skew / skew_se if skew_se > 0 else np.nan
        p_skew = 2 * stats.norm.sf(abs(skew_z)) if stats is not None and np.isfinite(skew_z) else np.nan
        results["null_skewness"] = {
            "hypothesis": "H0: skewness = 0",
            "statistic": skew_z,
            "p_value": p_skew,
            "reject_null": bool(stats is not None and np.isfinite(p_skew) and p_skew < 0.05),
            "note": "Approximate z-test for zero skewness."
        }

        kurtosis_se = np.sqrt(24 * n * (n - 1) ** 2 / ((n - 2) * (n - 3) * (n + 3) * (n + 5)))
        kurtosis_z = excess_kurtosis / kurtosis_se if kurtosis_se > 0 else np.nan
        p_kurtosis = 2 * stats.norm.sf(abs(kurtosis_z)) if stats is not None and np.isfinite(kurtosis_z) else np.nan
        results["null_kurtosis"] = {
            "hypothesis": "H0: excess kurtosis = 0",
            "statistic": kurtosis_z,
            "p_value": p_kurtosis,
            "reject_null": bool(stats is not None and np.isfinite(p_kurtosis) and p_kurtosis < 0.05),
            "note": "Approximate z-test for normal kurtosis."
        }

        jb_stat = (n / 6.0) * (skew ** 2 + (excess_kurtosis ** 2) / 4.0)
        p_jarque_bera = stats.chi2.sf(jb_stat, 2) if stats is not None else np.nan
        results["jarque_bera"] = {
            "hypothesis": "H0: data are normally distributed",
            "statistic": jb_stat,
            "p_value": p_jarque_bera,
            "reject_null": bool(stats is not None and p_jarque_bera < 0.05),
            "note": "Jarque-Bera test for normality."
        }

        for key, value in results.items():
            print(f"{key}: {value}")

        return results