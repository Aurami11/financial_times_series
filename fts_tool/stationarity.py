import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews


class StationarityTester:
    """Run stationarity and unit-root diagnostics on a time series."""

    def __init__(self, ts: pd.Series):
        if not isinstance(ts, pd.Series):
            raise TypeError("ts must be a pandas Series.")

        self.ts = ts

    def _clean_series(self) -> pd.Series:
        """Remove missing values while preserving the original index."""
        series = self.ts.dropna()

        if series.empty:
            raise ValueError("The series contains no observations after dropna().")

        return series

    @staticmethod
    def _validate_regression(regression: str, allowed: set[str]) -> None:
        if regression not in allowed:
            options = ", ".join(sorted(allowed))
            raise ValueError(
                f"Invalid regression={regression!r}. Choose from: {options}."
            )

    def adf_test(self, regression: str = "c") -> dict:
        """
        Augmented Dickey-Fuller test.

        H0: the series has a unit root.
        'c': constant; 'ct': constant and linear trend.
        """
        self._validate_regression(regression, {"c", "ct"})

        series = self._clean_series()
        result = adfuller(
            series,
            regression=regression,
            autolag="AIC"
        )

        return {
            "test": "ADF",
            "regression": regression,
            "statistic": result[0],
            "p_value": result[1],
            "lags_used": result[2],
            "n_obs": result[3],
            "critical_values": result[4],
        }

    def kpss_test(self, regression: str = "c") -> dict:
        """
        KPSS test.

        H0: the series is stationary around a level ('c')
        or a deterministic linear trend ('ct').
        """
        self._validate_regression(regression, {"c", "ct"})

        series = self._clean_series()
        result = kpss(
            series,
            regression=regression,
            nlags="auto"
        )

        return {
            "test": "KPSS",
            "regression": regression,
            "statistic": result[0],
            "p_value": result[1],
            "lags_used": result[2],
            "n_obs": len(series),
            "critical_values": result[3],
        }

    def zivot_andrews_test(self, regression: str = "c") -> dict:
        """
        Zivot-Andrews unit-root test allowing one structural break.

        'c': break in the intercept;
        't': break in the trend;
        'ct': break in both.
        """
        self._validate_regression(regression, {"c", "t", "ct"})

        series = self._clean_series()
        result = zivot_andrews(
            series,
            regression=regression,
            autolag="AIC",
        )
        break_position = result[4]

        return {
            "test": "Zivot-Andrews",
            "regression": regression,
            "statistic": result[0],
            "p_value": result[1],
            "critical_values": result[2],
            "lags_used": result[3],
            "n_obs": len(series),
            "break_position": break_position,
            "break_index": series.index[break_position],
        }