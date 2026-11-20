import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import METADATA_CSV_FILE, METRICS_FILE, PROCESSED_FILE
from src.predict import MODEL_FILES, load_ticker_metadata, predict_next_day_price


@st.cache_data
def load_company_metadata() -> dict:
    """Map ticker symbol to display name, exchange, currency, and quoteType."""
    if not METADATA_CSV_FILE.exists():
        return {}
    meta_df = pd.read_csv(METADATA_CSV_FILE)
    meta_df["symbol"] = meta_df["symbol"].astype(str).str.strip().str.upper()

    display_name = (
        meta_df["longName"]
        .fillna(meta_df.get("shortName"))
        .fillna(meta_df["symbol"])
        .astype(str)
        .str.strip()
    )

    exchange_col = (
        meta_df["fullExchangeName"].fillna(meta_df["exchange"])
        if "fullExchangeName" in meta_df.columns
        else meta_df["exchange"]
    )

    meta_dict = {}
    for idx, row in meta_df.iterrows():
        sym = row["symbol"]
        meta_dict[sym] = {
            "name": display_name.iloc[idx],
            "exchange": str(exchange_col.iloc[idx]) if pd.notna(exchange_col.iloc[idx]) else "N/A",
            "currency": str(row.get("currency", "N/A")) if pd.notna(row.get("currency")) else "N/A",
            "quoteType": str(row.get("quoteType", "N/A")) if pd.notna(row.get("quoteType")) else "N/A",
        }
    return meta_dict




st.set_page_config(page_title="Multi-Stock Price Predictor", page_icon="📈", layout="wide")

st.title("📈 Multi-Stock Price Prediction Engine")
st.caption("Global Scale-Invariant Regression across 1,000+ Equities | Linear, Polynomial & Decision Tree Models")

try:
    ticker_meta = load_ticker_metadata()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Sidebar: Company & Model Selection
st.sidebar.header("1. Select Company & Model")
company_meta = load_company_metadata()

tickers = sorted(
    ticker_meta.keys(),
    key=lambda t: company_meta.get(t, {}).get("name", t).lower(),
)

selected_ticker = st.sidebar.selectbox(
    "Company",
    options=tickers,
    index=0,
    format_func=lambda t: (
        f"{company_meta[t]['name']} ({t})" if t in company_meta else t
    ),
)
selected_model = st.sidebar.selectbox("Regression Model", list(MODEL_FILES.keys()), index=0)

info = company_meta.get(selected_ticker, {})
selected_label = f"{info['name']} ({selected_ticker})" if "name" in info else selected_ticker

defaults = ticker_meta[selected_ticker]
st.sidebar.markdown(
    f"**Last Data Date:** `{defaults['last_date']}`  \n"
    f"**Records Loaded:** `{defaults['rows']}`"
)

# Main Layout: Header + Metadata Badges
st.subheader(f"Input Parameters for {selected_label}")

if info:
    badge_style = (
        "display:inline-block; padding:3px 10px; margin-right:8px; "
        "border-radius:12px; font-size:0.82rem; font-weight:600;"
    )
    st.markdown(
        f"""
        <div style="margin-bottom: 12px;">
            <span style="{badge_style} background-color:#DBEAFE; color:#1E40AF;">
                Exchange: {info['exchange']}
            </span>
            <span style="{badge_style} background-color:#D1FAE5; color:#065F46;">
                Currency: {info['currency']}
            </span>
            <span style="{badge_style} background-color:#FEF3C7; color:#92400E;">
                Type: {info['quoteType']}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Main Layout: Parameter Inputs
st.write("Pre-filled with the latest trading day's values. Adjust any indicator to run a what-if estimate:")

col1, col2, col3 = st.columns(3)
with col1:
    open_price = st.number_input("Open ($)", value=float(defaults["open"]), min_value=0.01, step=0.1)
    high_price = st.number_input("High ($)", value=float(defaults["high"]), min_value=0.01, step=0.1)
    low_price = st.number_input("Low ($)", value=float(defaults["low"]), min_value=0.01, step=0.1)
    close_price = st.number_input("Close ($)", value=float(defaults["close"]), min_value=0.01, step=0.1)

with col2:
    ma_7 = st.number_input("7-Day Moving Avg ($)", value=float(defaults["ma_7"]), min_value=0.01, step=0.1)
    ma_30 = st.number_input("30-Day Moving Avg ($)", value=float(defaults["ma_30"]), min_value=0.01, step=0.1)
    daily_return = st.number_input("Daily Return (decimal)", value=float(defaults["daily_return"]), format="%.4f", step=0.005)

with col3:
    volatility_7 = st.number_input("7-Day Volatility", value=float(defaults["volatility_7"]), min_value=0.0, format="%.4f", step=0.002)
    volatility_30 = st.number_input("30-Day Volatility", value=float(defaults["volatility_30"]), min_value=0.0, format="%.4f", step=0.002)
    volume = st.number_input("Volume", value=int(defaults["volume"]), min_value=1, step=10000)
    volume_ma_7 = st.number_input("7-Day Avg Volume", value=float(defaults["volume_ma_7"]), min_value=1.0, step=10000.0)

if st.button("Predict Next-Day Close Price", type="primary"):
    result = predict_next_day_price(
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        close_price=close_price,
        volume=volume,
        ma_7=ma_7,
        ma_30=ma_30,
        daily_return=daily_return,
        volatility_7=volatility_7,
        volatility_30=volatility_30,
        volume_ma_7=volume_ma_7,
        model_name=selected_model,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Close", f"${close_price:,.2f}")
    m2.metric("Estimated Next-Day Close", f"${result['predicted_close']:,.2f}", delta=f"{result['price_delta']:+.2f} ({result['predicted_return_pct']:+.2f}%)")
    m3.metric("Model Used", selected_model)

st.divider()

# Historical Chart & Evaluation Table
tab1, tab2 = st.tabs(["Historical Price & Moving Averages", "Model Evaluation Metrics"])

with tab1:
    fallback_csv = PROCESSED_FILE.with_suffix(".csv")
    if PROCESSED_FILE.exists():
        hist_df = pd.read_parquet(PROCESSED_FILE)
    elif fallback_csv.exists():
        hist_df = pd.read_csv(fallback_csv, parse_dates=["Date"])
    else:
        hist_df = pd.DataFrame()
    if not hist_df.empty:
        ticker_hist = hist_df[hist_df["Ticker"] == selected_ticker].sort_values("Date")
        if not ticker_hist.empty:
            chart_df = ticker_hist.set_index("Date")[["Close", "MA_7", "MA_30"]]
            st.line_chart(chart_df)

with tab2:
    if METRICS_FILE.exists():
        st.dataframe(pd.read_csv(METRICS_FILE), use_container_width=True)
