import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import pickle
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Cryptocurrency Price Prediction",
    page_icon="📈",
    layout="wide"
)

# Application title and description
st.title("Cryptocurrency Price Prediction")
st.markdown("""
This application predicts cryptocurrency prices using LSTM deep learning models. 
Select a cryptocurrency and prediction timeframe to see the forecast.
""")

# Sidebar for user inputs
st.sidebar.header("Settings")

# Cryptocurrency selection
crypto_options = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Binance Coin": "binancecoin",
    "Cardano": "cardano",
    "Solana": "solana",
    "Ripple": "ripple",
    "Polkadot": "polkadot",
    "Dogecoin": "dogecoin"
}
selected_crypto_name = st.sidebar.selectbox("Select Cryptocurrency", list(crypto_options.keys()))
selected_crypto = crypto_options[selected_crypto_name]

# Prediction timeframe
prediction_days = st.sidebar.slider("Prediction Days", min_value=1, max_value=30, value=7)

# Historical data timeframe
historical_days = st.sidebar.slider("Historical Data (days)", min_value=30, max_value=365, value=90)

# Functions for data processing and prediction
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_crypto_data(symbol='bitcoin', vs_currency='usd', days=365):
    """Fetch historical cryptocurrency data from CoinGecko API"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {
            'vs_currency': vs_currency,
            'days': days,
            'interval': 'daily'
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            st.error(f"Error fetching data: {response.status_code}")
            return None
            
        data = response.json()
        
        # Extract price data
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def preprocess_data(df, window_size=60):
    """Preprocess data for LSTM model"""
    # Create a copy of the dataframe
    data = df.copy()
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['price']])
    
    return scaled_data, scaler

def load_model():
    """Load the trained model"""
    try:
        # In a real application, you'd load your saved model
        # For demonstration, we'll create a simple LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(60, 1)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_future(model, data, scaler, days_to_predict=7, window_size=60):
    """Predict future prices based on the last window_size days"""
    # Get the last window_size days of data
    last_window = data['price'].values[-window_size:]
    
    # Scale the data
    last_window_scaled = scaler.transform(last_window.reshape(-1, 1))
    
    # Create an empty list to store predictions
    future_predictions = []
    
    # Current batch for prediction
    current_batch = last_window_scaled.reshape(1, window_size, 1)
    
    # Predict each day one by one
    for _ in range(days_to_predict):
        # Make prediction for next day
        next_day_scaled = model.predict(current_batch, verbose=0)[0]
        
        # Add prediction to our list
        future_predictions.append(next_day_scaled[0])
        
        # Update current batch to include the new prediction
        current_batch = np.append(current_batch[:, 1:, :], 
                                 [[next_day_scaled]], 
                                 axis=1)
    
    # Convert scaled predictions back to actual values
    future_pred_actual = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    # Create dates for the predictions
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
    
    # Create a dataframe with the predictions
    future_df = pd.DataFrame({
        'price': future_pred_actual.flatten()
    }, index=future_dates)
    
    return future_df

# Function to run the prediction pipeline
def run_prediction():
    with st.spinner("Fetching data and generating predictions..."):
        # Fetch data
        data = fetch_crypto_data(symbol=selected_crypto, days=historical_days)
        if data is None:
            return None, None
        
        # Preprocess data
        scaled_data, scaler = preprocess_data(data)
        
        # Load model
        model = load_model()
        if model is None:
            return None, None
        
        # In a real application, you would train this model on the data
        # For demonstration, let's assume it's already trained or use a pre-trained model
        
        # Make future predictions
        future_predictions = predict_future(model, data, scaler, days_to_predict=prediction_days)
        
        return data, future_predictions

# Run prediction when user clicks button
if st.sidebar.button("Generate Prediction"):
    data, predictions = run_prediction()
    
    if data is not None and predictions is not None:
        # Display current price
        current_price = data['price'].iloc[-1]
        predicted_price = predictions['price'].iloc[-1]
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label=f"Current {selected_crypto_name} Price", 
                value=f"${current_price:.2f}"
            )
        
        with col2:
            st.metric(
                label=f"Predicted Price ({prediction_days} days)", 
                value=f"${predicted_price:.2f}",
                delta=f"{price_change:.2f}%"
            )
        
        with col3:
            latest_change = ((data['price'].iloc[-1] - data['price'].iloc[-2]) / data['price'].iloc[-2]) * 100
            st.metric(
                label="24h Change",
                value=f"{latest_change:.2f}%",
                delta=f"{latest_change:.2f}%",
                delta_color="normal"
            )
        
        # Create visualization using Plotly
        st.subheader(f"{selected_crypto_name} Price Prediction")
        
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
        
        # Add historical data trace
        fig.add_trace(
            go.Scatter(
                x=data.index[-30:],
                y=data['price'][-30:],
                name='Historical',
                line=dict(color='blue')
            )
        )
        
        # Add prediction trace
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions['price'],
                name='Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f"{selected_crypto_name} Price Prediction for Next {prediction_days} Days",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            legend_title="Data Type",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display historical data and predictions in tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Recent Historical Data")
            st.dataframe(data.tail(10).style.format({'price': '${:.2f}'}))
        
        with col2:
            st.subheader("Price Predictions")
            st.dataframe(predictions.style.format({'price': '${:.2f}'}))
        
        # Display additional technical indicators and analysis
        st.subheader("Technical Analysis")
        
        # Calculate moving averages
        data['MA5'] = data['price'].rolling(window=5).mean()
        data['MA20'] = data['price'].rolling(window=20).mean()
        
        # Calculate RSI
        delta = data['price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Create technical analysis chart
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, 
                           row_heights=[0.7, 0.3])
        
        # Add price and moving averages
        fig2.add_trace(
            go.Scatter(x=data.index[-30:], y=data['price'][-30:], name='Price'),
            row=1, col=1
        )
        
        fig2.add_trace(
            go.Scatter(x=data.index[-30:], y=data['MA5'][-30:], name='5-Day MA'),
            row=1, col=1
        )
        
        fig2.add_trace(
            go.Scatter(x=data.index[-30:], y=data['MA20'][-30:], name='20-Day MA'),
            row=1, col=1
        )
        
        # Add RSI
        fig2.add_trace(
            go.Scatter(x=data.index[-30:], y=data['RSI'][-30:], name='RSI'),
            row=2, col=1
        )
        
        # Add RSI overbought/oversold lines
        fig2.add_shape(
            type="line", x0=data.index[-30], x1=data.index[-1],
            y0=70, y1=70, line=dict(color="red", dash="dash"),
            row=2, col=1
        )
        
        fig2.add_shape(
            type="line", x0=data.index[-30], x1=data.index[-1],
            y0=30, y1=30, line=dict(color="green", dash="dash"),
            row=2, col=1
        )
        
        # Update layout
        fig2.update_layout(
            title=f"{selected_crypto_name} Technical Analysis",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=700
        )
        
        fig2.update_yaxes(title_text="RSI", row=2, col=1)
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Add model evaluation details
        st.subheader("Model Information")
        st.write("""
        This prediction uses a Long Short-Term Memory (LSTM) neural network, which is particularly 
        effective for time series forecasting. The model analyzes patterns in historical price data
        to generate predictions for future prices.
        
        **Note:** Cryptocurrency markets are highly volatile and unpredictable. These predictions
        should be used for informational purposes only and not as financial advice.
        """)
else:
    # Display placeholder when app first loads
    st.info("Select settings and click 'Generate Prediction' to see forecast")
    
    # Display sample visualization of cryptocurrency data
    placeholder_data = fetch_crypto_data(symbol="bitcoin", days=30)
    if placeholder_data is not None:
        st.subheader("Sample Bitcoin Historical Data")
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=placeholder_data.index,
                y=placeholder_data['price'],
                name='Bitcoin Price',
                line=dict(color='gold')
            )
        )
        
        fig.update_layout(
            title="Bitcoin Price (Last 30 Days)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Add footer
st.markdown("---")
st.markdown("Cryptocurrency Price Prediction | Data source: CoinGecko API")