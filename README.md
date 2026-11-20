# Multi-Stock Price Prediction Engine (Regression + Streamlit UI)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)

An end-to-end, production-structured machine learning project designed to ingest **1,000+ historical stock OHLCV CSV files**, engineer scale-invariant technical indicators, train and cross-validate regression models globally, and serve next-day price estimates through an interactive **Streamlit** frontend.

---

## Project Architecture

```text
stock-price-regression-ml/
├── app/
│   └── streamlit_app.py             # Interactive UI (company name selector, metadata badges, full-history chart)
├── data/
│   ├── metadata_stocks.csv          # Optional company metadata (symbol, longName, exchange, currency, quoteType)
│   ├── raw/                         # Drop all 1,000+ stock CSV files here (supports nested subfolders)
│   └── processed/                   # Combined engineered dataset (.parquet / .csv) & 5,000-row sample
├── models/                          # Serialized sklearn pipelines (.joblib) & ticker_metadata.json
├── reports/
│   ├── figures/                     # Predicted vs Actual, Model Comparison & Feature Importance plots
│   ├── model_metrics.csv            # MAE, MSE, RMSE, R² comparison table
│   └── model_evaluation_report.md   # Key indicators & short-term forecasting analysis
├── src/
│   ├── config.py                    # Centralized paths, features, and hyperparameters
│   ├── data_ingestion.py            # Recursive CSV loader, mixed date parser, missing value & outlier handling
│   ├── feature_engineering.py       # Per-ticker MA-7, MA-30, returns, volatility & scale-invariant ratios
│   ├── train.py                     # Chronological split, TimeSeriesSplit CV & global model training
│   ├── evaluate.py                  # Metrics computation & matplotlib visualizations
│   └── predict.py                   # Inference module used by the Streamlit app
├── tests/
│   └── test_pipeline.py             # Pytest suite for ingestion & feature validation
├── .gitignore
├── requirements.txt
└── README.md
```
---

## Key Features & Design Decisions
1. **Scale-Invariant Global Modeling:** Stock prices span $5 to $5,000+. Training directly on raw dollar prices causes high-priced stocks to dominate MSE. Instead, the pipeline engineers 9 relative, scale-invariant indicators (`Close_to_MA7, Close_to_MA30, MA7_to_MA30, Daily_Return, Intraday_Range, Open_Close_Return, Volatility_7, Volatility_30, Volume_Ratio_7`) to predict next-day percentage return, then reconstructs dollar price as `Close_t * (1 + predicted_return)`.

2. **Company-Agnostic Inference:** The company name/ticker is not used as a model feature—preventing 1,000-column one-hot sparsity and enabling zero-shot predictions on newly added stocks. Selecting a company in the UI pre-fills its latest trading-day parameters and filters its historical chart.

3. **Strict Per-Ticker Grouping:** Moving averages, rolling volatility, outlier capping, and target shifts (`shift(-1)`) run strictly inside `groupby("Ticker")` so adjacent CSV files never leak into one another.

4. **Recursive Ingestion & Mixed Date Support:** `src/data_ingestion.py` recursively discovers CSVs in `data/raw/` (including nested subfolders via rglob) and parses both `YYYY-MM-DD` and `DD-MM-YYYY` date formats using `format="mixed", dayfirst=True`.

5. **Zero Look-Ahead Bias:** Train/test splitting is performed chronologically per ticker (first 80% train, last 20% test), and cross-validation uses `TimeSeriesSplit`.

6. **Regularized Polynomial Regression:** Uses PolynomialFeatures(degree=2) paired with Ridge(alpha=10.0) inside a StandardScaler pipeline to prevent coefficient explosion across large multi-stock datasets.

7. **User-Friendly Streamlit UI:**

- Integrates `data/metadata_stocks.csv` to display searchable company names (`longName (symbol)`) sorted alphabetically, with automatic fallback (`longName` → `shortName` → `symbol`).

- Displays Exchange, Currency, and Quote Type header badges for the selected company.

- Plots the full historical timeline (`Close, MA_7, MA_30`) alongside the model evaluation table.
---


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
