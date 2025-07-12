# app.py
import streamlit as st
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import numpy as np
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from predictor import get_predictions
from io import BytesIO
import plotly.graph_objects as go

# 🌐 Page configuration
st.set_page_config(page_title="PulsePredict", layout="wide")
st.title("📈 PulsePredict - Real-Time Stock Price Forecast")

# 🔄 Auto-refresh every 60 seconds
st_autorefresh(interval=60000, key="refresh")
st.caption(f"⏱ Last updated: {datetime.datetime.now().strftime('%H:%M:%S')}")

with st.expander("ℹ️ About PulsePredict"):
    st.markdown("""
    **PulsePredict** is a real-time dashboard that uses machine learning to forecast stock prices.  
    It visualizes historical trends, predicted prices, trading volume, and technical indicators with an intuitive UI.
    """)

# 🎨 Theme toggle
st.sidebar.subheader("🌓 Theme Settings")
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
bg_color = 'white' if theme == "Light" else '#1e1e1e'
grid_color = '#ccc' if theme == "Light" else '#444'
label_color = 'black' if theme == "Light" else 'white'

# 📌 Company selection
companies = {
    "Apple (AAPL)": "AAPL", "Google (GOOGL)": "GOOGL", "Amazon (AMZN)": "AMZN",
    "Microsoft (MSFT)": "MSFT", "Tesla (TSLA)": "TSLA", "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX", "NVIDIA (NVDA)": "NVDA", "Adobe (ADBE)": "ADBE",
    "Intel (INTC)": "INTC"
}
company_name = st.selectbox("📌 Select a Company", list(companies.keys()))
ticker = companies[company_name]

# 💲 Current Price
try:
    stock_data = yf.Ticker(ticker)
    current_price = stock_data.history(period="1d")['Close'].iloc[-1]
    st.metric(f"💲 {ticker} Stock Price", f"${current_price:.2f}")
except Exception as e:
    st.warning(f"⚠ Could not fetch stock price: {e}")

# 📊 Helper for exporting plots
def fig_to_download(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf

# 📈 Model Predictions
try:
    result = get_predictions(ticker)

    # 📍 Technical indicators
    indicators = st.multiselect(
        "📊 Add Indicators:", ["MA20", "MA50", "Volume"], default=["MA20", "Volume"]
    )

    # 📈 Price Forecast Plot
    st.subheader("📈 Actual vs Predicted")
    fig1, ax1 = plt.subplots(figsize=(10, 5), facecolor=bg_color)
    ax1.plot(result["actual"], label="Actual", color="#1f77b4", linewidth=2)
    ax1.plot(result["predicted"], label="Predicted", color="#ff7f0e", linewidth=2)
    if "MA20" in indicators:
        ax1.plot(result["ma20"], label="MA 20", color="#2ca02c", linestyle="--")
    if "MA50" in indicators:
        ax1.plot(result["ma50"], label="MA 50", color="#9467bd", linestyle="--")
    ax1.set_xlabel("Days", color=label_color)
    ax1.set_ylabel("Price ($)", color=label_color)
    ax1.set_title("Forecast vs Actual", fontsize=14, fontweight='bold', color=label_color)
    ax1.grid(True, linestyle="--", alpha=0.3, color=grid_color)
    ax1.legend()
    st.pyplot(fig1)
    st.download_button("📥 Download Price Forecast", fig_to_download(fig1), f"{ticker}_forecast.png", "image/png")

    # 📉 Volume Plot
    if "Volume" in indicators:
        st.subheader("📉 Volume Over Time")
        fig2, ax2 = plt.subplots(figsize=(10, 3), facecolor=bg_color)
        volume = np.array(result["volume"]).flatten()
        ax2.bar(range(len(volume)), volume, color="#00bfa6", alpha=0.8, width=0.8)
        ax2.set_ylabel("Volume", color=label_color)
        ax2.set_title("Trading Volume", fontsize=14, fontweight='bold', color=label_color)
        ax2.grid(True, linestyle='--', alpha=0.3, color=grid_color)
        st.pyplot(fig2)
        st.download_button("📥 Download Volume Plot", fig_to_download(fig2), f"{ticker}_volume.png", "image/png")
    


    # 📊 Residual Error Histogram
    st.subheader("📊 Prediction Error Distribution")
    residuals = np.array(result["actual"]) - np.array(result["predicted"])
    fig3, ax3 = plt.subplots(figsize=(8, 4), facecolor=bg_color)
    n, bins, _ = ax3.hist(residuals, bins=30, color="#e377c2", alpha=0.75, density=True, edgecolor='white')
    mu, std = norm.fit(residuals)
    x = np.linspace(bins[0], bins[-1], 100)
    ax3.plot(x, norm.pdf(x, mu, std), 'k--', label='Normal Curve')
    ax3.set_xlabel("Error", color=label_color)
    ax3.set_ylabel("Density", color=label_color)
    ax3.set_title("Prediction Error Histogram", fontsize=14, fontweight='bold', color=label_color)
    ax3.grid(True, linestyle="--", alpha=0.3, color=grid_color)
    ax3.legend()
    st.pyplot(fig3)
    st.download_button("📥 Download Error Histogram", fig_to_download(fig3), f"{ticker}_errors.png", "image/png")

except Exception as e:
    st.error(f"⚠️ Something went wrong: {e}")
