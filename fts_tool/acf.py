import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
class ACF_Analysis:
    """
    A class for performing autocorrelation function (ACF) analysis on financial time series data.

    Attributes:
        ts (pd.Series): The financial time series data.
    """

    def __init__(self, ts):
        """
        Initializes the ACF_Analysis class with the provided financial time series data.

        Args:
            ts (pd.Series): The financial time series data.
        """
        self.ts = ts

    def compute_acf(self, nlags=40):
        """
        Computes the autocorrelation function (ACF) of the financial time series data.

        Args:
            nlags (int): The number of lags to compute the ACF for. Default is 40.

        Returns:
            pd.Series: A Series containing the computed ACF values.
        """
  
        return pd.Series(acf(self.ts, nlags=nlags), index=range(nlags + 1))
    
    def plot_acf(self, nlags=40, show_significance_interval=False, title="Autocorrelation Function (ACF)", xlabel="Lags", ylabel="ACF",
                 significance_level=0.05, figsize=(12, 6), ylim=None):
        """
        Plots the autocorrelation function (ACF) of the financial time series data.

        Args:
            nlags (int): The number of lags to plot the ACF for. Default is 40.
            show_significance_interval (bool): Whether to show dotted
                significance bounds. Default is False.
            significance_level (float): The two-sided significance level used
                for the bounds. Default is 0.05.
            figsize (tuple): The figure size as (width, height). Default is
                (12, 6).
            title (str): The title of the plot. Default is "Autocorrelation Function (ACF)".
            xlabel (str): The label for the x-axis. Default is "Lags".
            ylabel (str): The label for the y-axis. Default is "ACF".
            ylim (tuple): The y-axis limits as (ymin, ymax). Default is None,
                which lets Matplotlib choose the limits automatically.

        Returns:
            tuple: The Matplotlib figure and axes.
        """

        if not 0 < significance_level < 1:
            raise ValueError("significance_level must be between 0 and 1")

        fig = plot_acf(
            self.ts,
            lags=nlags,
            alpha=None if show_significance_interval else significance_level,
        )
        ax = fig.axes[0]
        if ylim is not None:
            ax.set_ylim(ylim)
        fig.set_size_inches(*figsize)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f7f9fc")
        ax.grid(axis="y", color="#d9e2ec", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9fb3c8")
        ax.spines["bottom"].set_color("#9fb3c8")
        ax.tick_params(colors="#486581")

        if show_significance_interval:
            critical_value = norm.ppf(1 - significance_level / 2)
            bound = critical_value / (len(self.ts) ** 0.5)
            interval_color = "#d64550"
            ax.axhline(bound, color=interval_color, linestyle=":", linewidth=1.5,
                       label=f"Intervalle {100 * (1 - significance_level):.0f}%")
            ax.axhline(-bound, color=interval_color, linestyle=":", linewidth=1.5)
            ax.legend(frameon=False, loc="upper right")

        ax.set_title(title, loc="left", pad=14, fontsize=15,
                     fontweight="bold", color="#102a43")
        ax.set_xlabel(xlabel, labelpad=8, color="#243b53")
        ax.set_ylabel(ylabel, labelpad=8, color="#243b53")
        fig.tight_layout()
        plt.show()

        return fig, ax

    def ljung_box_test(self, lags=10, graph=False):
        """
        Performs the Ljung-Box test for autocorrelation in the residuals of a time series.

        Args:
            lags (int): The number of lags to include in the test. Default is 10.
            graph (bool): Whether to plot the Ljung-Box test statistic. Default is False.

        Returns:
            pd.DataFrame: A DataFrame containing the Ljung-Box test statistics and p-values.
        """

        lb_test = acorr_ljungbox(self.ts, lags=lags, return_df=True)
        lb_test.columns = ["statistic", "p_value"]

        if graph:
            plt.figure(figsize=(10, 6))
            plt.plot(lb_test.index, lb_test["statistic"], marker='o', linestyle='-', color='b', label='Ljung-Box Statistic')
            plt.axhline(y=3.841, color='r', linestyle='--', label='Critical Value (0.05)')
            plt.title("Ljung-Box Test Statistic")
            plt.xlabel("Lag")
            plt.ylabel("Statistic")
            plt.legend()
            plt.grid()
            plt.show()

        return lb_test