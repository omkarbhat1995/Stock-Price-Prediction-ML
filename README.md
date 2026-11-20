# Multi-Stock Price Prediction Engine (Regression + Streamlit UI)

An end-to-end, production-structured machine learning project designed to ingest **1,000+ historical stock OHLCV CSV files**, engineer scale-invariant technical indicators, train and cross-validate regression models globally, and serve next-day price estimates through an interactive **Streamlit** frontend.

---

## Project Architecture

```text
stock-price-regression-ml/
├── app/
│   └── streamlit_app.py             # Interactive frontend (company selector + parameter tuning)
├── data/
│   ├── raw/                         # Drop all 1,000+ stock CSV files here (e.g., IOT.csv, AAPL.csv)
│   └── processed/                   # Engineered dataset (.parquet / .csv)
├── models/                          # Serialized sklearn pipelines (.joblib) & ticker_metadata.json
├── reports/
│   ├── figures/                     # Predicted vs Actual, Model Comparison & Feature Importance plots
│   ├── model_metrics.csv            # MAE, MSE, RMSE, R² comparison table
│   └── model_evaluation_report.md   # Key indicators & short-term forecasting analysis
├── src/
│   ├── config.py                    # Centralized paths, features, and hyperparameters
│   ├── data_ingestion.py            # Batch CSV loader, missing value & outlier handling per ticker
│   ├── feature_engineering.py       # Per-ticker MA-7, MA-30, returns, volatility & scale-invariant ratios
│   ├── train.py                     # Chronological split, TimeSeriesSplit CV & model training
│   ├── evaluate.py                  # Metrics computation & matplotlib visualizations
│   └── predict.py                   # Inference module used by the Streamlit app
├── tests/
│   └── test_pipeline.py             # Pytest suite for ingestion & feature validation
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Why This Scales Cleanly to 1,000+ Stocks

1. **Scale-Invariant Global Modeling:** Stock prices span $5 to $5,000+. Training directly on raw dollar prices causes high-priced stocks to dominate MSE. Instead, the pipeline engineers **relative ratios** (`Close/MA_7 - 1`, `Close/MA_30 - 1`, `MA_7/MA_30 - 1`, intraday range, log volume ratio) to predict **next-day percentage return**, then reconstructs dollar price as `Close_t * (1 + predicted_return)`.
2. **Strict Per-Ticker Grouping:** Moving averages, rolling volatility, and target shifts (`shift(-1)`) are computed inside `groupby("Ticker")` so adjacent CSV files never leak into one another.
3. **Zero Look-Ahead Bias:** Train/test splitting is performed **chronologically per ticker** (first 80% train, last 20% test), and cross-validation uses `TimeSeriesSplit`.
4. **Regularized Polynomial Regression:** Uses `PolynomialFeatures(degree=2)` paired with `Ridge` regularization inside a `StandardScaler` pipeline to prevent coefficient explosion across large multi-stock datasets.

---

## Quick Start

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add Your Stock CSV Files
Place up to 1,000+ Yahoo Finance-format CSV files (`Date, Open, High, Low, Close, Volume`) inside `data/raw/`. Each filename (e.g., `AAPL.csv`, `IOT.csv`) is automatically parsed as the company ticker.

### 3. Run the Training & Evaluation Pipeline
```bash
python -m src.train
```
*(Optional: test on a subset first using `python -m src.train --max-files 50`)*

This generates:
- `data/processed/engineered_stocks.parquet` (or `.csv` fallback)
- `models/*.joblib` & `models/ticker_metadata.json`
- `reports/model_metrics.csv`
- `reports/figures/*.png`

### 4. Launch the Frontend
```bash
streamlit run app/streamlit_app.py
```

### 5. Run Tests
```bash
pytest
```

---

## Sample Benchmark Results (`IOT.csv` Single-Stock Baseline)

| Model | CV MAE (Return) | Test MAE ($) | Test MSE ($²) | Test RMSE ($) | Test R² |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Baseline (`Close_t`) | — | 0.4815 | 0.3144 | 0.5607 | 0.7471 |
| Decision Tree Regressor | 0.0542 | 0.8725 | 0.9832 | 0.9916 | 0.2091 |
| Linear Regression | 0.0767 | 0.8849 | 1.0388 | 1.0192 | 0.1643 |
| Polynomial Regression (Deg 2) | 0.0662 | 1.0379 | 1.4490 | 1.2037 | -0.1657 |

See `reports/model_evaluation_report.md` for a detailed breakdown of feature importance, regime shift, and short-term forecasting limitations.
