import json
import joblib
import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, MODELS_DIR, TICKER_METADATA_FILE

MODEL_FILES = {
    "Linear Regression": "linear_regression.joblib",
    "Polynomial Regression (Deg 2)": "polynomial_regression_deg_2.joblib",
    "Decision Tree Regressor": "decision_tree_regressor.joblib",
    "Best Model": "best_model.joblib",
}


def load_ticker_metadata() -> dict:
    if not TICKER_METADATA_FILE.exists():
        raise FileNotFoundError("Ticker metadata not found. Run `python -m src.train` first.")
    with open(TICKER_METADATA_FILE, "r") as f:
        return json.load(f)


def load_model(model_name: str = "Best Model"):
    filename = MODEL_FILES.get(model_name, "best_model.joblib")
    model_path = MODELS_DIR / filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Run `python -m src.train` first.")
    return joblib.load(model_path)


def predict_next_day_price(
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    volume: float,
    ma_7: float,
    ma_30: float,
    daily_return: float,
    volatility_7: float,
    volatility_30: float,
    volume_ma_7: float,
    model_name: str = "Best Model",
) -> dict:
    """Compute engineered features from user inputs and predict next-day close price."""
    model = load_model(model_name)

    features = pd.DataFrame([{
        "Close_to_MA7": (close_price / ma_7) - 1.0 if ma_7 > 0 else 0.0,
        "Close_to_MA30": (close_price / ma_30) - 1.0 if ma_30 > 0 else 0.0,
        "MA7_to_MA30": (ma_7 / ma_30) - 1.0 if ma_30 > 0 else 0.0,
        "Daily_Return": daily_return,
        "Intraday_Range": (high_price - low_price) / close_price if close_price > 0 else 0.0,
        "Open_Close_Return": (close_price - open_price) / open_price if open_price > 0 else 0.0,
        "Volatility_7": volatility_7,
        "Volatility_30": volatility_30,
        "Volume_Ratio_7": float(np.log1p(volume / volume_ma_7)) if volume_ma_7 > 0 else 0.0,
    }])[FEATURE_COLUMNS]

    pred_return = float(model.predict(features)[0])
    # Clip extreme single-day predictions for stability
    pred_return = float(np.clip(pred_return, -0.25, 0.25))
    pred_close = close_price * (1.0 + pred_return)

    return {
        "predicted_close": round(pred_close, 2),
        "predicted_return_pct": round(pred_return * 100, 2),
        "price_delta": round(pred_close - close_price, 2),
    }
