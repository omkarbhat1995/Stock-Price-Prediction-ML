import glob
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR, REQUIRED_COLUMNS, OUTLIER_Z_THRESHOLD


def load_single_stock(file_path: Path) -> Optional[pd.DataFrame]:
    """Load and validate a single stock CSV file."""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[WARN] Skipping {file_path.name}: failed to read ({e})")
        return None

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    if not all(col in df.columns for col in REQUIRED_COLUMNS):
        print(f"[WARN] Skipping {file_path.name}: missing required columns")
        return None

    df = df[REQUIRED_COLUMNS].copy()
    raw_dates = df["Date"].astype(str).str.strip()
    parsed_dates = None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"):
        attempt = pd.to_datetime(raw_dates, format=fmt, errors="coerce")
        if attempt.notna().mean() > 0.95:
            parsed_dates = attempt
            break
    df["Date"] = parsed_dates if parsed_dates is not None else pd.to_datetime(raw_dates, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"])

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle missing values using time-safe forward fill then backward fill
    df[["Open", "High", "Low", "Close", "Volume"]] = (
        df[["Open", "High", "Low", "Close", "Volume"]].ffill().bfill()
    )

    # Filter invalid non-positive prices
    df = df[(df["Close"] > 0) & (df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0)]
    if len(df) < 40:
        print(f"[WARN] Skipping {file_path.name}: fewer than 40 valid rows ({len(df)})")
        return None

    # Cap extreme daily return outliers (e.g., unadjusted splits or data glitches)
    pct_change = df["Close"].pct_change()
    mean_ret = pct_change.mean()
    std_ret = pct_change.std()
    if std_ret and std_ret > 0:
        z_scores = (pct_change - mean_ret).abs() / std_ret
        outlier_mask = z_scores > OUTLIER_Z_THRESHOLD
        if outlier_mask.any():
            # Winsorize Volume spikes and clip extreme price jumps via rolling median
            rolling_med = df["Close"].rolling(window=5, min_periods=1, center=True).median()
            df.loc[outlier_mask, "Close"] = rolling_med[outlier_mask]

    # Cap extreme volume outliers at 99th percentile per ticker
    vol_99 = df["Volume"].quantile(0.99)
    if vol_99 > 0:
        df["Volume"] = np.minimum(df["Volume"], vol_99)

    df["Ticker"] = file_path.stem.upper()
    return df.reset_index(drop=True)


def load_all_stocks(raw_dir: Path = RAW_DATA_DIR, max_files: Optional[int] = None) -> pd.DataFrame:
    """Batch load all stock CSV files from raw_dir into a single DataFrame."""
    csv_files = sorted(raw_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    if max_files is not None:
        csv_files = csv_files[:max_files]

    frames: List[pd.DataFrame] = []
    for path in csv_files:
        stock_df = load_single_stock(path)
        if stock_df is not None:
            frames.append(stock_df)

    if not frames:
        raise ValueError("No valid stock datasets were loaded.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Loaded {len(frames)} ticker(s) | Total rows: {len(combined):,}")
    return combined
