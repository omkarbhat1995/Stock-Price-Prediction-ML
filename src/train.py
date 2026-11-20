import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    CV_SPLITS,
    FEATURE_COLUMNS,
    METRICS_FILE,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_CLOSE_COL,
    TARGET_RETURN_COL,
    TEST_SIZE_RATIO,
)
from src.data_ingestion import load_all_stocks
from src.feature_engineering import build_feature_dataset
from src.evaluate import (
    compute_metrics,
    plot_feature_importance,
    plot_model_comparison,
    plot_predicted_vs_actual,
)


def chronological_split_per_ticker(df: pd.DataFrame, test_ratio: float = TEST_SIZE_RATIO):
    """Split each ticker chronologically (no look-ahead bias)."""
    train_parts, test_parts = [], []
    for _, group in df.groupby("Ticker", sort=False):
        group = group.sort_values("Date")
        split_idx = int(len(group) * (1 - test_ratio))
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True).sort_values("Date").reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sort_values("Date").reset_index(drop=True)
    return train_df, test_df


def build_models() -> dict:
    """Define regression pipelines with feature scaling."""
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Polynomial Regression (Deg 2)": Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", Ridge(alpha=10.0)),  # Ridge prevents degree-2 variance explosion across 1,000 stocks
        ]),
        "Decision Tree Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=RANDOM_STATE)),
        ]),
    }


def train_and_evaluate(max_files: int = None):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_all_stocks(max_files=max_files)
    df = build_feature_dataset(raw_df, save=True)

    train_df, test_df = chronological_split_per_ticker(df)
    X_train, y_train_ret = train_df[FEATURE_COLUMNS], train_df[TARGET_RETURN_COL]
    X_test, y_test_ret = test_df[FEATURE_COLUMNS], test_df[TARGET_RETURN_COL]
    y_test_close = test_df[TARGET_CLOSE_COL].values
    base_close_test = test_df["Close"].values

    models = build_models()
    tscv = TimeSeriesSplit(n_splits=min(CV_SPLITS, max(2, len(train_df) // 20)))

    results = []
    price_predictions = {}
    trained_pipelines = {}

    # Naive Persistence Baseline (Tomorrow's Close == Today's Close)
    baseline_metrics = compute_metrics(y_test_close, base_close_test)
    results.append({
        "Model": "Naive Baseline (Close_t)",
        "CV_MAE_Return": np.nan,
        "Test_MAE_Price": baseline_metrics["MAE"],
        "Test_MSE_Price": baseline_metrics["MSE"],
        "Test_RMSE_Price": baseline_metrics["RMSE"],
        "Test_R2_Price": baseline_metrics["R2"],
    })

    for name, pipeline in models.items():
        cv_scores = cross_val_score(
            pipeline, X_train, y_train_ret, cv=tscv, scoring="neg_mean_absolute_error"
        )
        cv_mae_ret = float(-cv_scores.mean())

        pipeline.fit(X_train, y_train_ret)
        pred_ret = pipeline.predict(X_test)

        # Reconstruct predicted next-day Close price from predicted return
        pred_close = base_close_test * (1.0 + pred_ret)
        price_metrics = compute_metrics(y_test_close, pred_close)

        results.append({
            "Model": name,
            "CV_MAE_Return": round(cv_mae_ret, 6),
            "Test_MAE_Price": price_metrics["MAE"],
            "Test_MSE_Price": price_metrics["MSE"],
            "Test_RMSE_Price": price_metrics["RMSE"],
            "Test_R2_Price": price_metrics["R2"],
        })

        price_predictions[name] = pred_close
        trained_pipelines[name] = pipeline

        # Save each trained pipeline
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(pipeline, MODELS_DIR / f"{safe_name}.joblib")

    metrics_df = pd.DataFrame(results).sort_values("Test_MAE_Price")
    metrics_df.to_csv(METRICS_FILE, index=False)

    # Save best model pointer
    ml_only = metrics_df[metrics_df["Model"] != "Naive Baseline (Close_t)"]
    best_model_name = ml_only.iloc[0]["Model"]
    joblib.dump(trained_pipelines[best_model_name], MODELS_DIR / "best_model.joblib")

    # Generate visual deliverables
    plot_predicted_vs_actual(test_df, price_predictions)
    plot_model_comparison(metrics_df)
    plot_feature_importance(
        trained_pipelines["Linear Regression"],
        trained_pipelines["Decision Tree Regressor"],
    )

    print("\n=== Model Evaluation Summary ===")
    print(metrics_df.to_string(index=False))
    print(f"\n[INFO] Best ML model saved: {best_model_name}")
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-stock regression models.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit on CSV files to load.")
    args = parser.parse_args()
    train_and_evaluate(max_files=args.max_files)
