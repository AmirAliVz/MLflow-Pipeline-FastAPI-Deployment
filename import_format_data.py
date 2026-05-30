import logging
import os
import pandas as pd

verbose = False

# Target schema column order (exact names for production model / MLflow)
SCHEMA_COLUMNS = [
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "ORG_AIRPORT",
    "DEST_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "DEPARTURE_TIME",
    "DEPARTURE_DELAY",
    "SCHEDULED_ARRIVAL",
    "ARRIVAL_TIME",
    "ARRIVAL_DELAY",
]

# Rename map: support both underscore and camelCase input names
RENAME_MAP = {
    "DAY_OF_MONTH": "DAY",
    "DAY": "DAY",
    "ORIGIN": "ORG_AIRPORT",
    "DEST": "DEST_AIRPORT",
    "CRS_DEP_TIME": "SCHEDULED_DEPARTURE",
    "CRSDepTime": "SCHEDULED_DEPARTURE",
    "DEP_TIME": "DEPARTURE_TIME",
    "DepTime": "DEPARTURE_TIME",
    "DEP_DELAY": "DEPARTURE_DELAY",
    "DepDelay": "DEPARTURE_DELAY",
    "CRS_ARR_TIME": "SCHEDULED_ARRIVAL",
    "CRSArrTime": "SCHEDULED_ARRIVAL",
    "ARR_TIME": "ARRIVAL_TIME",
    "ArrTime": "ARRIVAL_TIME",
    "ARR_DELAY": "ARRIVAL_DELAY",
    "ArrDelay": "ARRIVAL_DELAY",
}

# Logging setup: write to logs/ for MLflow artifact pickup
os.makedirs("logs", exist_ok=True)
_logger = logging.getLogger("import_format_data")
_logger.setLevel(logging.INFO)
_logger.handlers.clear()
_logger.propagate = False
_fh = logging.FileHandler("logs/import_format_data_log.txt", mode="w")
_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
_logger.addHandler(_fh)
if verbose:
    _logger.addHandler(logging.StreamHandler())

# Read dataset into a DataFrame
df = pd.read_csv("data/T_ONTIME_REPORTING.csv")
df_clean = df.copy()

_logger.info("=" * 50)
_logger.info(" MISSING VALUE SUMMARY (before formatting) ")
_logger.info("=" * 50)
_logger.info("--> Number of Missing Values per Column:")
_logger.info("\n%s", df_clean.isnull().sum())

# ----- 1. Column renames -----
# Only rename columns that exist (handles underscore vs camelCase sources)
rename_applied = {k: v for k, v in RENAME_MAP.items() if k in df_clean.columns}
df_clean = df_clean.rename(columns=rename_applied)

# Ensure exactly the 12 schema columns in order (drop any extras, order correctly)
df_clean = df_clean[[c for c in SCHEMA_COLUMNS if c in df_clean.columns]]
df_clean = df_clean.reindex(columns=SCHEMA_COLUMNS)

# ----- 2. Coerce types (keep NaNs for dropna) -----
numeric_columns = [c for c in SCHEMA_COLUMNS if c not in ("ORG_AIRPORT", "DEST_AIRPORT")]
for col in numeric_columns:
    if col not in df_clean.columns:
        continue
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

# Keep string columns as string
for col in ("ORG_AIRPORT", "DEST_AIRPORT"):
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype(str)

# ----- 3. Drop rows with null values -----
_logger.info("\n--> DataFrame size before dropping nulls: %s rows", len(df_clean))

df_clean = df_clean.dropna()

_logger.info("--> DataFrame size after dropping nulls: %s rows", len(df_clean))

# Convert numeric columns to integer (no nulls left)
for col in numeric_columns:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype(int)

_logger.info("=" * 50)
_logger.info(" AFTER FORMATTING ")
_logger.info("=" * 50)
_logger.info("--> Shape: %s", df_clean.shape)
_logger.info("--> Dtypes:\n%s", df_clean.dtypes)
_logger.info("--> Missing per column:\n%s", df_clean.isnull().sum())

# ----- Save output (input for filter_clean_data.py) -----
df_clean.to_csv("data/formatted_data.csv", index=False)
_logger.info("--> Saved: formatted_data.csv")

logging.shutdown()