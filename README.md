# Financial Time Series

A practical and evolving collection of notes, notebooks, experiments, and tools inspired by *Analysis of Financial Time Series* by Ruey S. Tsay.

This repository serves as both a personal learning journal and a technical reference for applying financial time-series methods to real market data. The goal is to turn theoretical concepts into reproducible analyses, visualisations, and reusable Python utilities.

## Motivation

Financial time series offer an intuitive framework for modelling asset prices, returns, volatility, dependence, and risk. Through this project, I explore the statistical foundations of financial modelling while building practical implementations in Python.

Rather than treating the book as purely theoretical material, each topic is approached through:

- Real financial-market datasets
- Exploratory data analysis and visualisation
- Statistical modelling and diagnostics
- Interpretation of results in a financial context
- Reusable utilities packaged into a dedicated Python toolkit

## Notebooks

Each notebook focuses on a specific chapter or subject area and aims to connect theory with concrete market applications.

For example, `01_characteristics.ipynb` explores the main characteristics of financial time series through assets such as Apple (`AAPL`):

- Price and return visualisation
- Simple and logarithmic return computation
- Distributional analysis of returns
- Normal-distribution fitting
- Histograms, density plots, and Q–Q plots
- Initial observations on skewness, kurtosis, and heavy tails
- Interpretation of stylised facts in financial markets

The notebooks are designed to be exploratory, reproducible, and progressively more rigorous as the project develops.

## Python Toolkit

Alongside the notebooks, this repository includes a Python toolkit intended to make analyses more consistent, modular, and reusable.

The package will progressively provide utilities for:

- Downloading and managing market data
- Computing simple and log returns
- Cleaning and transforming time-series data
- Descriptive statistics and distribution diagnostics
- Stationarity and autocorrelation analysis
- Volatility modelling
- Time-series visualisation
- Backtesting and performance evaluation
- Portfolio and risk-related analytics

The objective is to avoid duplicating code across notebooks and to create a clean foundation for future financial-research projects.

## Technologies

- Python
- Jupyter Notebook
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`
- `statsmodels`
- `arch`
- `yfinance` or other market-data providers

## Getting Started

Clone the repository:

```bash
git clone https://github.com/aurami11/financial-time-series.git
cd financial-time-series
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Then open the notebook corresponding to the topic you want to explore.

## Disclaimer

This repository is an educational and research-oriented project. Nothing in this repository should be interpreted as financial advice, investment advice, or a recommendation to buy or sell any financial instrument.