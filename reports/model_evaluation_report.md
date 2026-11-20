# Multi-Stock Price Regression — Evaluation & Insights Report

## 1. Dataset & Preprocessing Summary
- **Universe Loaded:** **996 equities** ingested from `data/raw/`, totaling **5,890,747 raw OHLCV records**.
- **Quality Filtering:** 4 tickers were automatically excluded during ingestion for having fewer than 40 valid trading days (`HKD.csv`: 16 rows, `HLN.csv`: 11 rows, `INTR.csv`: 32 rows, `PSNYW.csv`: 31 rows).
- **Engineered Dataset:** **5,829,532 usable rows** (**98.96% retained**; 61,215 rows dropped due to the 30-day rolling moving average warm-up and next-day target shift), saved to `data/processed/engineered_stocks.parquet`.
- **Cleaning & Scaling:** Strict whole-column date format parsing (`%Y-%m-%d`, `%d-%m-%Y`, `%m-%d-%Y`), per-ticker chronological sorting, duplicate date removal, time-safe forward/backward fill, Z-score return outlier capping (`|z| > 4.0`), 99th-percentile volume winsorization, and `StandardScaler` normalization inside `scikit-learn` `Pipeline` objects.

## 2. Key Stock Indicators & Feature Impact
Comparing **Linear Regression** (absolute standardized coefficients) and **Decision Tree Regressor** (variance-reduction importance) across 5.83M rows (`reports/figures/feature_importance.png`) reveals that **medium-term trend deviation** dominates next-day return modeling:

1. **Linear Regression Top Drivers:**
   - **`Close_to_MA30` (~0.072)** and **`MA7_to_MA30` (~0.066)** are by far the strongest predictors, capturing price extension relative to the 30-day trend and 7-day/30-day crossover momentum.
   - **`Open_Close_Return` (~0.017)** is the third strongest signal, reflecting intraday directional conviction.
   - Secondary contributors include **`Volume_Ratio_7` (~0.006)**, **`Close_to_MA7` (~0.004)**, and **`Intraday_Range` (~0.002)**.
   - **`Daily_Return`**, **`Volatility_7`**, and **`Volatility_30`** carry near-zero direct linear weight.
2. **Decision Tree Regressor Top Driver:**
   - **`Close_to_MA30`** accounts for virtually **100% of split importance** at `max_depth=5`, indicating that non-linear tree splits focus almost entirely on bounding extreme deviations from the 30-day moving average while keeping typical-day predictions conservative and stable.

## 3. Model Comparison & Error Analysis (996-Stock Chronological Test Split)

| Model | CV MAE (Return) | Test MAE ($) | Test MSE ($²) | Test RMSE ($) | Test R² |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Baseline (`Close_t`) | — | 1.2992 | 10.7449 | 3.2779 | 0.9992 |
| Decision Tree Regressor | 0.0170 | 1.3015 | 13.5673 | 3.6834 | 0.9990 |
| Linear Regression | 0.0170 | 2.1234 | 27.6515 | 5.2585 | 0.9980 |
| Polynomial Regression (Deg 2) | 0.0984 | 5.5484 | 722.3737 | 26.8770 | 0.9485 |

### Key Takeaways from the Metrics & Plots
- **Best ML Model — Decision Tree Regressor:** Achieved the lowest cross-validated return error (**CV MAE = 0.01695**, ~1.70% average daily return error) and lowest ML test error (**Test MAE = $1.3015**, **RMSE = $3.6834**, **$R^2$ = 0.9990**), trailing the Naive Persistence Baseline (`$1.2992`) by only **$0.0023 (+0.18%)**. In `reports/figures/actual_vs_predicted_RTX.png`, the Decision Tree tracks actual prices tightly across a 10-year test span (2012–2022), including the 2020 market sell-off and subsequent recovery.
- **Linear Regression Stability vs. Tail Sensitivity:** Linear Regression nearly matched the Decision Tree in return CV (**0.01698**), but its dollar **Test MAE ($2.1234)** and **RMSE ($5.2585)** were higher because unbounded linear extrapolation produces larger dollar errors on high-priced stocks during volatile sessions.
- **Polynomial Regression (Deg 2) Instability:** Even with `Ridge(alpha=10.0)` regularization, Polynomial Regression degraded sharply (**CV MAE = 0.0984**, **Test MAE = $5.5484**, **Test MSE = 722.3737**, **RMSE = $26.8770**, **$R^2$ = 0.9485**). As seen in `actual_vs_predicted_RTX.png`, quadratic interaction terms amplify extreme standardized feature values on volatile days, producing severe upward and downward prediction spikes (e.g., overshooting above $130 in 2015 and below $15 in 2019–2020).
- **Random-Walk Benchmark Confirmation:** The Naive Baseline (`Close_t`) slightly edged out all three ML models on Test MAE (`$1.2992`) and Test RMSE (`$3.2779`). This aligns with financial theory (weak-form market efficiency): daily equity closing prices closely follow a random walk, making `Close_t` an exceptionally strong benchmark.

## 4. Usefulness for Short-Term Predictions
- **Where the Model Adds Value:**
  - **Interactive Scenario & Sensitivity Analysis:** In the Streamlit frontend, users can adjust moving averages, volume spikes, or intraday range to see how short-term technical signals shift next-day return expectations across 996 companies.
  - **Mean-Reversion & Trend Screening:** Both Linear and Decision Tree models confirm that extreme `Close_to_MA30` and `MA7_to_MA30` divergences are the most informative OHLCV signals.
  - **Scalable Architecture:** The scale-invariant return formulation generalizes across 5.83M rows and arbitrary stock price scales without retraining per ticker.
- **Limitations & Recommendations:**
  - High price-level $R^2$ (`0.9990`) is primarily driven by strong day-to-day price autocorrelation rather than alpha generation.
  - Pure OHLCV technical indicators cannot anticipate exogenous events (earnings surprises, interest-rate decisions, M&A news).
  - **Next Steps for Production Use:** Treat predictions as **quantitative decision-support signals** rather than standalone trading triggers. Future iterations should incorporate non-OHLCV features (sector/macro indicators, earnings calendars, news sentiment) and evaluate regularized gradient-boosted trees (e.g., XGBoost/LightGBM) with Huber loss to further suppress tail errors.