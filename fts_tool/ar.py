from statsmodels.tsa.stattools import pacf
import matplotlib.pyplot as plt

class AR_Analysis:
   """
   This class is used for AutoRegressive (AR) analysis of financial time series data.
   """

   def __init__(self, ts) :
      """
      Initializes the AR_Analysis with the provided time series data.

      Args:
         ts (pd.Series): A pandas Series containing the time series data for analysis.
      """
      self.ts = ts

   def compute_pacf(self, nlags: int = 20):
      """
      Computes the Partial Autocorrelation Function (PACF) for the time series data.

      Args:
         nlags (int): The number of lags to compute the PACF for.
      """

      pacf_values = pacf(self.ts, nlags=nlags)

      return pacf_values

   def plot_pacf(self, nlags: int = 20, title: str = "Partial Autocorrelation Function (PACF)", xlabel: str = "Lags", ylabel: str = "PACF", figsize: tuple = (12, 6), ylim: tuple = None):
      """
      Plots the Partial Autocorrelation Function (PACF) for the time series data.

      Args:
         nlags (int): The number of lags to plot the PACF for.
         title (str): The title of the plot.
         xlabel (str): The label for the x-axis.
         ylabel (str): The label for the y-axis.
         figsize (tuple): The size of the figure as (width, height).
         ylim (tuple): The limits for the y-axis as (ymin, ymax).
      """

      pacf_values = self.compute_pacf(nlags=nlags)

      fig, ax = plt.subplots(figsize=figsize)
      ax.bar(range(len(pacf_values)), pacf_values, color='blue', alpha=0.7)
      ax.axhline(y=0, color='black', linewidth=0.8)
      ax.set_title(title)
      ax.set_xlabel(xlabel)
      ax.set_ylabel(ylabel)

      if ylim is not None:
         ax.set_ylim(ylim)

      plt.show()

   def criteron_selection(self, max_lag: int = 20, full_output: bool = False):
      """
      Determines the optimal lag order for the AR model using AIC and BIC criteria.

      Args:
         max_lag (int): The maximum number of lags to consider for the AR model.
         full_output (bool): If True, returns the AIC and BIC values for all lags. Default is False.
      """

      aic_values = []
      bic_values = []

      for lag in range(1, max_lag + 1):
         model = sm.tsa.AR(self.ts).fit(maxlag=lag)
         aic_values.append(model.aic)
         bic_values.append(model.bic)

      optimal_aic_lag = aic_values.index(min(aic_values)) + 1
      optimal_bic_lag = bic_values.index(min(bic_values)) + 1

      if full_output:
         return optimal_aic_lag, optimal_bic_lag, aic_values, bic_values

      return optimal_aic_lag, optimal_bic_lag