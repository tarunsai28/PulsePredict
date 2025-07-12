import yfinance as yf
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

def get_predictions(ticker):
    df = yf.download(ticker, period='1y', interval='1d')
    data = df[['Close', 'Volume']].dropna()
    close_prices = data['Close'].values.reshape(-1, 1)

    # Calculate indicators
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    # Prepare LSTM sequences
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])
    X = np.array(X)
    y = np.array(y)

    model_path = f"model_{ticker.upper()}.h5"

    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)
        model.save(model_path)

    predicted = model.predict(X)
    predicted_prices = scaler.inverse_transform(predicted)
    actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

    recent_data = data.iloc[-len(predicted_prices):]

    return {
        "actual": actual_prices.flatten(),
        "predicted": predicted_prices.flatten(),
        "ma20": recent_data['MA20'].values,
        "ma50": recent_data['MA50'].values,
        "volume": recent_data['Volume'].values,
        
    }
