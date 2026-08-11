"""This module contains configuration settings for the Financial Time Series (FTS) tool. It defines various parameters and options that can be adjusted to customize the behavior of the tool, including data sources, processing options, and storage settings. """

import os
import logging
from pathlib import Path

# Define default data storage directory
DEFAULT_DATA_DIR = Path(os.getenv("FTS_DATA_DIR", "./data"))

# Define prices data source
PRICES_DATA_SOURCE = DEFAULT_DATA_DIR / "prices"