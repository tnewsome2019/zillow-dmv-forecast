
import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

# Streamlit App
st.set_page_config(page_title="Zillow DMV Forecast", page_icon="🏡", layout="wide")
st.title("🏡 Zillow DMV Market Forecast")
st.markdown("### Predict Median Sale Prices in the DMV Region!")
st.write("Forecast housing prices for D.C., Maryland, and Virginia with Zillow data.")

# Sidebar
st.sidebar.header("DMV Options")
available_states = ["District of Columbia", "Maryland", "Virginia"]
state = st.sidebar.selectbox("Select DMV Area", available_states, index=0)
show_history = st.sidebar.checkbox("Show Historical Trend", value=True)

# Load model
state_key = state.lower().replace(" ", "_")
model_file = f"rf_model_{state_key}.pkl"

if os.path.exists(model_file):
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    st.write(f"✅ Loaded model for {state}")
else:
    st.error(f"Model file '{model_file}' not found. Please add it to the folder.")
    st.stop()

# Load history
data_file = f"{state.replace(' ', '_')}_history.csv"
try:
    history_df = pd.read_csv(data_file)
    history_df['Date'] = pd.to_datetime(history_df['Date'])
except FileNotFoundError:
    st.warning(f"History file '{data_file}' not found. Historical trend unavailable.")
    history_df = pd.DataFrame()

# Input
st.subheader("Set Your Input")
lag_price = st.slider(
    "Last Month’s Median Sale Price ($)",
    min_value=int(history_df['Sale_Prices'].min()) if not history_df.empty else 100000,
    max_value=int(history_df['Sale_Prices'].max()) if not history_df.empty else 1500000,
    value=int(history_df['Sale_Prices'].iloc[-1]) if not history_df.empty else 500000,
    step=10000,
    format="$%d"
)

# Prediction
input_data = pd.DataFrame([[lag_price]], columns=['LagPrice'])
prediction = model.predict(input_data)[0]
conf_low, conf_high = prediction * 0.9, prediction * 1.1

# Display forecast
st.subheader("Forecast Result")
col1, col2 = st.columns(2)
with col1:
    st.metric("Last Month", f"${lag_price:,.0f}")
with col2:
    st.metric("Next Month Prediction", f"${prediction:,.0f}", delta=f"${prediction - lag_price:,.0f}")
st.write(f"Confidence Range: ${conf_low:,.0f} - ${conf_high:,.0f}")

# Historical trend
if show_history and not history_df.empty:
    st.subheader(f"{state} Price Trend")
    plot_df = pd.concat([
        history_df.tail(5),
        pd.DataFrame({'Date': [history_df['Date'].iloc[-1], pd.Timestamp('2025-04-01')], 'Sale_Prices': [lag_price, prediction]})
    ])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_df['Date'], plot_df['Sale_Prices'], marker='o', color='#1f77b4', label='Actual + Forecast')
    ax.axvline(history_df['Date'].iloc[-1], color='gray', linestyle='--', alpha=0.5, label='Forecast Start')
    ax.fill_between([plot_df['Date'].iloc[-2], plot_df['Date'].iloc[-1]], [lag_price, conf_low], [lag_price, conf_high], color='#1f77b4', alpha=0.1, label='Confidence')
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    st.pyplot(fig)

st.markdown("---")
st.write("✨ Forecast generated on March 14, 2025. Data from Zillow via kagglehub.")
