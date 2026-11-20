import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import FIGURES_DIR, FEATURE_COLUMNS


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics: MAE, MSE, RMSE, and R2."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": round(float(mae), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
    }


def plot_predicted_vs_actual(test_df: pd.DataFrame, predictions: dict, ticker: str = None):
    """Plot actual vs predicted stock prices over time for a representative ticker."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if ticker is None:
        ticker = test_df["Ticker"].value_counts().index[0]

    mask = test_df["Ticker"] == ticker
    sub = test_df.loc[mask].sort_values("Date")

    plt.figure(figsize=(12, 6))
    plt.plot(sub["Date"], sub["Target_Next_Close"], label="Actual Next-Day Close", color="#111827", linewidth=2.2)

    colors = ["#2563EB", "#D97706", "#059669", "#DC2626"]
    for idx, (model_name, preds) in enumerate(predictions.items()):
        sub_preds = pd.Series(preds, index=test_df.index).loc[sub.index]
        plt.plot(
            sub["Date"],
            sub_preds,
            label=model_name,
            linestyle="--",
            linewidth=1.6,
            color=colors[idx % len(colors)],
        )

    plt.title(f"Actual vs Predicted Next-Day Close Price ({ticker})", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = FIGURES_DIR / f"actual_vs_predicted_{ticker}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_model_comparison(metrics_df: pd.DataFrame):
    """Bar chart comparing MAE and R2 across models."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(metrics_df["Model"], metrics_df["Test_MAE_Price"], color="#2563EB")
    axes[0].set_title("Test MAE (Price $) — Lower is Better", fontweight="bold")
    axes[0].set_ylabel("MAE ($)")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(metrics_df["Model"], metrics_df["Test_R2_Price"], color="#059669")
    axes[1].set_title("Test R² Score (Price) — Higher is Better", fontweight="bold")
    axes[1].set_ylabel("R²")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "model_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_feature_importance(linear_pipeline, tree_pipeline):
    """Plot standardized linear coefficients and Decision Tree feature importances."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Linear Regression absolute standardized coefficients
    lr_coefs = np.abs(linear_pipeline.named_steps["model"].coef_)
    lr_idx = np.argsort(lr_coefs)
    axes[0].barh([FEATURE_COLUMNS[i] for i in lr_idx], lr_coefs[lr_idx], color="#2563EB")
    axes[0].set_title("Linear Regression |Standardized Coef|", fontweight="bold")
    axes[0].set_xlabel("Magnitude")
    axes[0].grid(axis="x", alpha=0.3)

    # Decision Tree feature importances
    dt_imp = tree_pipeline.named_steps["model"].feature_importances_
    dt_idx = np.argsort(dt_imp)
    axes[1].barh([FEATURE_COLUMNS[i] for i in dt_idx], dt_imp[dt_idx], color="#D97706")
    axes[1].set_title("Decision Tree Feature Importance", fontweight="bold")
    axes[1].set_xlabel("Importance")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "feature_importance.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
