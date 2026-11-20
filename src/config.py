from pathlib import Path

# Directory Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METADATA_CSV_FILE = ROOT_DIR / "data" / "metadata_stocks.csv"
PROCESSED_FILE = PROCESSED_DATA_DIR / "engineered_stocks.parquet"
PROCESSED_CSV_SAMPLE = PROCESSED_DATA_DIR / "engineered_stocks_sample.csv"
METRICS_FILE = REPORTS_DIR / "model_metrics.csv"
TICKER_METADATA_FILE = MODELS_DIR / "ticker_metadata.json"

# Modeling Configuration
REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

# Scale-invariant & raw features for multi-stock generalization
FEATURE_COLUMNS = [
    "Close_to_MA7",
    "Close_to_MA30",
    "MA7_to_MA30",
    "Daily_Return",
    "Intraday_Range",
    "Open_Close_Return",
    "Volatility_7",
    "Volatility_30",
    "Volume_Ratio_7",
]

# Target: Next-day percentage return (reconstructed to next-day Close price)
TARGET_RETURN_COL = "Target_Next_Return"
TARGET_CLOSE_COL = "Target_Next_Close"

# Split & Cross-Validation
TEST_SIZE_RATIO = 0.2
CV_SPLITS = 5
RANDOM_STATE = 42
OUTLIER_Z_THRESHOLD = 4.0
