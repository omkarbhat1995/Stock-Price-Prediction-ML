import json
import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    PROCESSED_CSV_SAMPLE,
    PROCESSED_DATA_DIR,
    PROCESSED_FILE,
    TARGET_CLOSE_COL,
    TARGET_RETURN_COL,
    TICKER_METADATA_FILE,
    MODELS_DIR,
)


def engineer_features_for_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators and targets strictly within a single ticker."""
    df = df.sort_values("Date").copy()

    # Moving Averages (7-day and 30-day)
    df["MA_7"] = df["Close"].rolling(window=7, min_periods=7).mean()
    df["MA_30"] = df["Close"].rolling(window=30, min_periods=30).mean()

    # Daily Returns & Rolling Volatility
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility_7"] = df["Daily_Return"].rolling(window=7, min_periods=7).std()
    df["Volatility_30"] = df["Daily_Return"].rolling(window=30, min_periods=30).std()

    # Scale-Invariant Ratios (enables 1 global model across 1,000+ stocks at different price scales)
    df["Close_to_MA7"] = (df["Close"] / df["MA_7"]) - 1.0
    df["Close_to_MA30"] = (df["Close"] / df["MA_30"]) - 1.0
    df["MA7_to_MA30"] = (df["MA_7"] / df["MA_30"]) - 1.0
    df["Intraday_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Open_Close_Return"] = (df["Close"] - df["Open"]) / df["Open"]

    # Volume Ratio (log-stabilized relative to 7-day average volume)
    vol_ma7 = df["Volume"].rolling(window=7, min_periods=7).mean().replace(0, np.nan)
    df["Volume_MA_7"] = vol_ma7
    df["Volume_Ratio_7"] = np.log1p(df["Volume"] / vol_ma7)

    # Prediction Target: Next Trading Day's Close and Return
    df[TARGET_CLOSE_COL] = df["Close"].shift(-1)
    df[TARGET_RETURN_COL] = (df[TARGET_CLOSE_COL] - df["Close"]) / df["Close"]

    return df


def build_feature_dataset(raw_df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Apply feature engineering per ticker and persist processed data & metadata."""
    engineered_parts = []
    ticker_meta = {}

    for ticker, group in raw_df.groupby("Ticker", sort=True):
        feat_df = engineer_features_for_ticker(group)
        # Save latest known row per ticker for the frontend defaults
        valid_latest = feat_df.dropna(subset=["MA_7", "MA_30", "Volatility_7", "Volatility_30", "Volume_MA_7"])
        if not valid_latest.empty:
            last_row = valid_latest.iloc[-1]
            ticker_meta[ticker] = {
                "last_date": str(last_row["Date"].date()),
                "open": round(float(last_row["Open"]), 4),
                "high": round(float(last_row["High"]), 4),
                "low": round(float(last_row["Low"]), 4),
                "close": round(float(last_row["Close"]), 4),
                "volume": int(last_row["Volume"]),
                "ma_7": round(float(last_row["MA_7"]), 4),
                "ma_30": round(float(last_row["MA_30"]), 4),
                "daily_return": round(float(last_row["Daily_Return"]), 6),
                "volatility_7": round(float(last_row["Volatility_7"]), 6),
                "volatility_30": round(float(last_row["Volatility_30"]), 6),
                "volume_ma_7": round(float(last_row["Volume_MA_7"]), 2),
                "rows": int(len(group)),
            }
        engineered_parts.append(feat_df)

    full_df = pd.concat(engineered_parts, ignore_index=True)
    clean_df = full_df.dropna(subset=FEATURE_COLUMNS + [TARGET_RETURN_COL, TARGET_CLOSE_COL]).reset_index(drop=True)

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            clean_df.to_parquet(PROCESSED_FILE, index=False)
            saved_path = PROCESSED_FILE
        except ImportError:
            fallback_csv = PROCESSED_DATA_DIR / "engineered_stocks.csv"
            clean_df.to_csv(fallback_csv, index=False)
            saved_path = fallback_csv
        clean_df.head(5000).to_csv(PROCESSED_CSV_SAMPLE, index=False)
        with open(TICKER_METADATA_FILE, "w") as f:
            json.dump(ticker_meta, f, indent=2)
        print(f"[INFO] Saved engineered dataset: {saved_path} ({len(clean_df):,} rows)")

    return clean_df
