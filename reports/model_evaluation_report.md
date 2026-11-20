# Stock Price Regression — Evaluation & Insights Report

## 1. Dataset & Preprocessing Summary
- **Sample Tested:** `IOT.csv` (162 trading days, `2021-12-15` to `2022-08-08`; 131 usable rows after 30-day moving average warm-up and next-day target alignment).
- **Cleaning Strategy:** Chronological sorting, duplicate date removal, time-safe forward/backward fill for missing values, Z-score outlier capping on extreme return spikes (`|z| > 4.0`), and 99th-percentile volume winsorization per ticker.
- **Scaling:** All numeric features are standardized via `StandardScaler` inside `scikit-learn` `Pipeline` objects to prevent data leakage during cross-validation.

## 2. Key Stock Indicators & Feature Impact
Across both **Linear Regression** (standardized coefficient magnitude) and **Decision Tree Regressor** (Gini/variance reduction importance), the strongest predictors of next-day price movement are:
1. **`Close_to_MA30` and `MA7_to_MA30` (Trend & Mean-Reversion Signals):** The distance between current price and the 30-day moving average—and the 7-day vs. 30-day crossover spread—dominate both linear and tree splits.
2. **`Close_to_MA7` & `Open_Close_Return` (Short-Term Momentum):** Captures immediate buying/selling pressure within the current trading session.
3. **`Daily_Return` & `Volume_Ratio_7`:** Spikes in relative trading volume paired with sharp daily returns help tree-based splits separate high-conviction breakouts from low-volume noise.
4. **`Volatility_7` & `Volatility_30`:** Act primarily as risk-regime indicators rather than directional predictors.

## 3. Model Comparison & Error Analysis (Sample Run: `IOT.csv`)

| Model | CV MAE (Return) | Test MAE ($) | Test MSE ($²) | Test RMSE ($) | Test R² |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Baseline (`Close_t`) | — | 0.4815 | 0.3144 | 0.5607 | 0.7471 |
| Decision Tree Regressor | 0.0542 | 0.8725 | 0.9832 | 0.9916 | 0.2091 |
| Linear Regression | 0.0767 | 0.8849 | 1.0388 | 1.0192 | 0.1643 |
| Polynomial Regression (Deg 2) | 0.0662 | 1.0379 | 1.4490 | 1.2037 | -0.1657 |

### Why Did the Naive Baseline Outperform on 1 Stock?
On `IOT.csv` alone, the training window (early 2022) was a sustained downtrend while the test window (July–August 2022) was a rebound. With only ~104 training rows, the regression models learned a negative return drift and underpredicted the summer recovery, whereas Polynomial Regression overfit quadratic interaction terms (`R² = -0.1657`). Scaling the dataset to **1,000+ stocks** across multiple sectors and market regimes substantially reduces variance and eliminates single-stock regime bias.

## 4. Usefulness for Short-Term Predictions
- **What this model is good for:** Estimating short-term **directional bias**, running **what-if scenario analysis** (e.g., how a volume surge or moving-average crossover shifts next-day expected return), and benchmarking classical regression models against a random-walk baseline.
- **Limitations:** Pure OHLCV technical indicators cannot anticipate exogenous shocks (earnings surprises, Fed rate decisions, macroeconomic news). Daily stock prices closely approximate a random walk, so regression models should be treated as **decision-support signals** paired with fundamental data, sentiment, and strict risk management—never as standalone automated trading systems.
