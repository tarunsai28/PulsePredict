# 📈 PulsePredict

**PulsePredict** is a real-time machine learning dashboard that forecasts stock prices using an LSTM-based deep learning model. Built with Streamlit, it provides an interactive and visually rich interface for stock trend analysis, technical indicators, and error diagnostics.

---

## 🚀 Features

- 🔮 **Price Forecasting** using LSTM models (trained per stock)
- 📊 **Historical Price Visualization** with technical indicators (MA20, MA50)
- 📉 **Volume Over Time** with customizable bar charts
- 🎨 **Theme Switcher** (Light/Dark Mode)
- 🖼 **Exportable Charts** (Price, Volume, Error Distribution)
- 🧠 **Error Analysis** with histogram and normal distribution overlay
- 🔄 **Auto-refresh** for near real-time updates
- 💬 **Tooltips & Indicators** for better UX

---

## 🧪 Demo

> Screenshots or hosted link (Streamlit Cloud, if deployed)

---

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, TensorFlow (LSTM), yfinance
- **Visualization**: Matplotlib, Plotly, Seaborn
- **ML**: Keras Sequential LSTM model

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/tarunsai28/PulsePredict.git
cd PulsePredict

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

📁 Project Structure
bash
Copy
Edit
PulsePredict/
├── app.py                 # Main Streamlit app
├── predictor.py           # ML model loader and prediction logic
├── model/
│   └── train_model.py     # (Optional) Script to train/update models
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md

📌 Notes
Models are auto-trained if not found (model_<TICKER>.h5)

Default ticker data is fetched from Yahoo Finance

Make sure internet connection is available for live price updates

📄 License
MIT License

👤 Author
Tarun Sai Tirumala
🔗 tarunsai28

yaml
Copy
Edit
