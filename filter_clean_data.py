"""
Read formatted_data.csv (output of import_format_data.py), apply airport filter
and cleaning steps, save cleaned_data.csv (input for poly_regressor_Python_1.0.0.py).
"""
import logging
import os
import pandas as pd

verbose = False

# Chosen origin airport to keep (e.g., JFK)
CHOSEN_AIRPORT = "JFK"

# Extreme delay threshold (minutes): drop rows with |delay| > this
MAX_DELAY_MINUTES = 1000

# Logging setup: write to logs/ for MLflow artifact pickup
os.makedirs("logs", exist_ok=True)
_logger = logging.getLogger("filter_clean_data")
_logger.setLevel(logging.INFO)
_logger.handlers.clear()
_logger.propagate = False
_fh = logging.FileHandler("logs/filter_clean_data_log.txt", mode="w")
_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
_logger.addHandler(_fh)
if verbose:
    _logger.addHandler(logging.StreamHandler())

# Read formatted data (output of import_format_data.py)
df = pd.read_csv("data/formatted_data.csv")

_logger.info("=" * 50)
_logger.info(" BEFORE CLEANING ")
_logger.info("=" * 50)
_logger.info("--> Shape: %s", df.shape)
_logger.info("--> Missing per column:\n%s", df.isnull().sum())
_logger.info("--> Columns: %s", list(df.columns))


# 1. Filter for chosen airport (e.g., JFK)
df = df[df["ORG_AIRPORT"] == CHOSEN_AIRPORT]

# Cleaning steps
# 2. Drop duplicates
df = df.drop_duplicates()

# 3. Remove extreme delay values
df = df[
    (df["DEPARTURE_DELAY"].abs() <= MAX_DELAY_MINUTES)
    & (df["ARRIVAL_DELAY"].abs() <= MAX_DELAY_MINUTES)
]

_logger.info("\n" + "=" * 50)
_logger.info(" AFTER CLEANING ")
_logger.info("=" * 50)
_logger.info("--> Shape: %s", df.shape)
_logger.info("--> Dtypes:\n%s", df.dtypes)
_logger.info("--> Missing per column:\n%s", df.isnull().sum())

# Save output (input for poly_regressor_Python_1.0.0.py)
df.to_csv("data/cleaned_data.csv", index=False)
_logger.info("--> Saved: cleaned_data.csv")

logging.shutdown()
