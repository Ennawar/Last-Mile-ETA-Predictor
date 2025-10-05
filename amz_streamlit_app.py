# ==============================================================================
# AMZ Project: Streamlit Prediction Interface (app.py)
# ==============================================================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from math import radians, sin, cos, sqrt, asin

# --- 1. Load Model and Data for Reference ---
try:
    # Load the entire pipeline (preprocessor + model)
    model = joblib.load('best_model_pipeline.pkl')
    # Use the original data for dropdown lists in the app
    df_cleaned = pd.read_csv('amazon_delivery_cleaned.csv')
    st.success("Trained Model and Data Loaded Successfully!")
except FileNotFoundError:
    st.error("Required files ('best_model_pipeline.pkl' or 'amazon_delivery_cleaned.csv') not found. Please ensure they are in the same folder.")
    st.stop()


# --- 2. Helper Function (Must match the one used in training) ---

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates Haversine distance in km."""
    R = 6371  # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat/2.0)**2 + cos(lat1) * cos(lat2) * sin(dlon/2.0)**2
    c = 2 * asin(sqrt(a))
    
    return R * c

# --- 3. Streamlit Application Interface ---

st.title("📦 AMZ Delivery Time Predictor")
st.markdown("Estimate the delivery time (in minutes) based on order, agent, and environmental factors.")

with st.form("prediction_form"):
    st.header("1. Agent and Location Details")
    col1, col2 = st.columns(2)
    
    # Agent Details 
    agent_age = col1.slider("Agent Age", 18, 50, 30)
    agent_rating = col2.slider("Agent Rating", 2.0, 5.0, 4.5, 0.1)
    
    # Coordinates 
    st.subheader("Store and Drop-off Coordinates")
    st_lat = col1.number_input("Store Latitude", value=12.9728, format="%.4f")
    st_lon = col2.number_input("Store Longitude", value=80.2500, format="%.4f")
    dr_lat = col1.number_input("Drop Latitude", value=13.0128, format="%.4f")
    dr_lon = col2.number_input("Drop Longitude", value=80.2900, format="%.4f")
    
    st.header("2. Order and Environment Details")
    col3, col4, col5 = st.columns(3)
    
    # Environment and Order Info
    weather = col3.selectbox("Weather Condition", sorted(df_cleaned['Weather'].unique()))
    traffic = col4.selectbox("Traffic Condition", sorted(df_cleaned['Traffic'].unique()))
    category = col5.selectbox("Product Category", sorted(df_cleaned['Category'].unique()))
    
    vehicle = col3.selectbox("Delivery Vehicle", sorted(df_cleaned['Vehicle'].unique()))
    area = col4.selectbox("Delivery Area", sorted(df_cleaned['Area'].unique()))
    
    st.subheader("Time Details")
    order_hour = col5.slider("Order Hour (24h)", 0, 23, 15)
    order_dayofweek = col3.slider("Order Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
    
    # Time to Pickup is a derived feature
    time_to_pickup_min = col4.slider("Time from Order to Pickup (min)", 0, 60, 15)
    
    # Submit Button
    submitted = st.form_submit_button("Predict Delivery Time")

# --- 4. Prediction Logic ---

if submitted:
    
    # 1. Calculate derived feature: Delivery_Distance_km
    delivery_distance_km = haversine_distance(st_lat, st_lon, dr_lat, dr_lon)
    
    # 2. Create DataFrame for prediction (must match the training feature structure)
    input_data = pd.DataFrame({
        'Agent_Age': [agent_age],
        'Agent_Rating': [agent_rating],
        'Weather': [weather],
        'Traffic': [traffic],
        'Vehicle': [vehicle],
        'Area': [area],
        'Delivery_Distance_km': [delivery_distance_km],
        'Time_to_Pickup_min': [time_to_pickup_min],
        'Order_Hour': [order_hour],
        'Order_DayOfWeek': [order_dayofweek],
        'Category': [category],
    })
    
    # --- Make Prediction ---
    try:
        # Use the loaded pipeline to preprocess and predict
        prediction = model.predict(input_data)[0]
        
        st.success(f"**Predicted Delivery Time:**")
        st.balloons()
        
        # Display the result
        st.metric("Estimated Delivery Time", f"{prediction:.0f} minutes")
        st.info(f"The calculation was based on a **{delivery_distance_km:.2f} km** delivery distance.")
        
    except Exception as e:
        st.error(f"An error occurred during prediction. Please check your inputs: {e}")