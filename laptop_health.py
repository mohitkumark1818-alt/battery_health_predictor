import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("battery_health_model.pkl")

st.title("Laptop Battery Health Prediction")

# User inputs
charging_frequency_per_day = st.number_input(
    "Charging Frequency Per Day", min_value=0.0, max_value=20.0, value=2.0
)

avg_temperature_c = st.number_input(
    "Average Temperature (°C)", min_value=0.0, max_value=100.0, value=40.0
)

heavy_load_hours_per_week = st.number_input(
    "Heavy Load Hours Per Week", min_value=0.0, max_value=168.0, value=10.0
)

avg_charge_level_percent = st.number_input(
    "Average Charge Level (%)", min_value=0.0, max_value=100.0, value=70.0
)

gaming_hours_per_week = st.number_input(
    "Gaming Hours Per Week", min_value=0.0, max_value=168.0, value=5.0
)

battery_cycles = st.number_input(
    "Battery Cycles", min_value=0.0, value=300.0
)

battery_type = st.selectbox(
    "Battery Type",
    ["Li-ion", "Li-Po"]
)

fast_charging_usage_percent = st.number_input(
    "Fast Charging Usage (%)", min_value=0.0, max_value=100.0, value=20.0
)

avg_daily_usage_hours = st.number_input(
    "Average Daily Usage Hours", min_value=0.0, max_value=24.0, value=6.0
)

laptop_age_months = st.number_input(
    "Laptop Age (Months)", min_value=0.0, value=24.0
)

usage_pattern = st.selectbox(
    "Usage Pattern",
    ["light", "moderate", "heavy"]
)


# IMPORTANT:
# These column names MUST exactly match the columns used during training.
input_data = pd.DataFrame([{
    "charging_frequency_per_day": charging_frequency_per_day,
    "avg_temperature_c": avg_temperature_c,
    "heavy_load_hours_per_week": heavy_load_hours_per_week,
    "avg_charge_level_percent": avg_charge_level_percent,
    "gaming_hours_per_week": gaming_hours_per_week,
    "battery_cycles": battery_cycles,
    "battery_type": battery_type,
    "fast_charging_usage_percent": fast_charging_usage_percent,
    "avg_daily_usage_hours": avg_daily_usage_hours,
    "laptop_age_months": laptop_age_months,
    "usage_pattern": usage_pattern
}])


if st.button("Predict Battery Health"):

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Battery Health: {prediction:.2f}%")
