import pandas as pd
from src.data_ingestion import load_single_stock
from src.feature_engineering import engineer_features_for_ticker
from src.config import RAW_DATA_DIR, FEATURE_COLUMNS


def test_sample_ingestion_and_features():
    sample_path = RAW_DATA_DIR / "IOT.csv"
    df = load_single_stock(sample_path)
    assert df is not None
    assert len(df) > 40
    assert df.isnull().sum().sum() == 0

    feat_df = engineer_features_for_ticker(df)
    for col in FEATURE_COLUMNS:
        assert col in feat_df.columns
