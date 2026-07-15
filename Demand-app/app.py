import streamlit as st
import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Configure the visual layout of the Streamlit page
st.set_page_config(
    page_title="Demand Forecasting AI",
    page_icon="🍦",
    layout="centered"
)

# @st.cache_resource tells Streamlit to only train the AI ONCE when the app starts, 
# instead of retraining it every time the user clicks a button. This makes the app lightning fast!
@st.cache_resource
def load_and_train_ai():
    # 1. Download data from Hugging Face
    dataset = load_dataset("ashish-soni08/ice-cream-demand", split="train") 
    df = dataset.to_pandas()
    
    # 2. Clean the data
    df = df.dropna()
    
    # 3. Separate Features (X) and Target (y)
    X = df[['Temperature', 'Rainfall']]
    y = df['IceCreamsSold']
    
    # 4. Train the AI Brain (Switched to Linear Regression!)
    model = LinearRegression()
    model.fit(X, y)
    
    return model

st.title("🍦 AI Demand Forecaster")
st.write("Predict tomorrow's ice cream inventory needs based on the weather forecast.")
st.markdown("---")

# Show a loading spinner while the AI trains for the first time
with st.spinner('Training AI model on historical data...'):
    ai_model = load_and_train_ai()

st.subheader("Step 1: Enter Tomorrow's Weather Forecast")

# Create two side-by-side columns for the inputs
col1, col2 = st.columns(2)

with col1:
    # Input box for Temperature
    temp_input = st.number_input(
        "Temperature (°C)", 
        min_value=-10.0, 
        max_value=50.0, 
        value=30.0, 
        step=1.0,
        help="Expected average temperature tomorrow."
    )

with col2:
    # Input box for Rainfall
    rain_input = st.number_input(
        "Rainfall (mm)", 
        min_value=0.0, 
        max_value=100.0, 
        value=0.0, 
        step=1.0,
        help="Expected rainfall tomorrow in millimeters."
    )

st.markdown("---")

st.subheader("Step 2: Generate Prediction")

# When the user clicks this button, the code inside the block runs
if st.button("🔮 Predict Inventory Needed", use_container_width=True):
    
    # --- NEW: DATA PREPROCESSING STEP ---
    # The AI was trained on US data. We must convert the user's Metric inputs 
    # (Celsius/mm) into Imperial units (Fahrenheit/inches) before asking the AI!
    temp_f = (temp_input * 9/5) + 32
    rain_inches = rain_input / 25.4
    
    # Format the CONVERTED inputs into a DataFrame so the AI can read it
    user_data = pd.DataFrame({
        'Temperature': [temp_f], 
        'Rainfall': [rain_inches]
    })
    
    # Ask the AI for the prediction
    prediction = ai_model.predict(user_data)[0]
    
    # Round it to a whole number and ensure it NEVER drops below 0 using max()
    final_prediction = max(0, int(round(prediction)))
    
    # Display the result beautifully using a Streamlit Metric widget
    st.success("AI Prediction Complete!")
    st.metric(
        label="Recommended Ice Cream Stock", 
        value=f"{final_prediction} Units",
        delta="Based on weather correlation"
    )