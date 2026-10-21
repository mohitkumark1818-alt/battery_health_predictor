import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Laptop Battery Health Predictor",
    page_icon=None,
    layout="wide"
)

# Load trained pipeline model
@st.cache_resource
def load_model():
    return joblib.load("battery_health_model.pkl")

model = load_model()

st.title("Laptop Battery Health Predictor")
st.markdown(
    "Estimate your laptop's remaining battery health percentage based on usage patterns, "
    "thermal conditions, and charging habits using a trained Random Forest regression model."
)

st.divider()

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Hardware & Cycling Telemetry")
    battery_cycles = st.number_input(
        "Battery Cycles Completed", min_value=0.0, max_value=5000.0, value=300.0, step=10.0,
        help="Total number of full charge-discharge cycles recorded."
    )
    laptop_age_months = st.number_input(
        "Laptop Age (Months)", min_value=0.0, max_value=120.0, value=24.0, step=1.0,
        help="How many months you have owned or operated the laptop."
    )
    battery_type = st.selectbox(
        "Battery Chemistry Type",
        ["Li-ion", "Li-polymer"],
        help="Select the internal cell chemistry of your battery."
    )
    avg_temperature_c = st.number_input(
        "Average Operating Temperature (°C)", min_value=10.0, max_value=100.0, value=38.0, step=0.5,
        help="Typical battery package temperature during regular use."
    )
    charging_frequency_per_day = st.number_input(
        "Charging Frequency (Times / Day)", min_value=0.0, max_value=20.0, value=2.0, step=0.5,
        help="How many times the device is connected to power per day."
    )

with col2:
    st.subheader("Workload & Charging Behavior")
    avg_daily_usage_hours = st.number_input(
        "Average Daily Usage (Hours / Day)", min_value=0.0, max_value=24.0, value=6.0, step=0.5,
        help="Total uptime hours per day."
    )
    avg_charge_level_percent = st.number_input(
        "Average State of Charge (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0,
        help="Typical charge percentage maintained throughout the day."
    )
    fast_charging_usage_percent = st.number_input(
        "Fast Charging Usage (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0,
        help="Percentage of charging cycles performed via high-wattage fast chargers."
    )
    heavy_load_hours_per_week = st.number_input(
        "Heavy Load Hours (Hours / Week)", min_value=0.0, max_value=168.0, value=8.0, step=1.0,
        help="Time spent weekly on intensive tasks like rendering, compilation, or machine learning."
    )
    gaming_hours_per_week = st.number_input(
        "Gaming Hours (Hours / Week)", min_value=0.0, max_value=168.0, value=5.0, step=1.0,
        help="Weekly hours dedicated to gaming workloads."
    )
    usage_pattern = st.selectbox(
        "Overall Workflow Profile",
        ["Light", "Moderate", "Heavy", "Very Heavy"],
        index=1,
        help="Subjective characterization of your average usage intensity."
    )

st.divider()

if st.button("Predict Battery Health", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "battery_cycles": battery_cycles,
        "laptop_age_months": laptop_age_months,
        "avg_daily_usage_hours": avg_daily_usage_hours,
        "avg_charge_level_percent": avg_charge_level_percent,
        "charging_frequency_per_day": charging_frequency_per_day,
        "gaming_hours_per_week": gaming_hours_per_week,
        "avg_temperature_c": avg_temperature_c,
        "fast_charging_usage_percent": fast_charging_usage_percent,
        "heavy_load_hours_per_week": heavy_load_hours_per_week,
        "battery_type": battery_type,
        "usage_pattern": usage_pattern
    }])

    prediction = float(model.predict(input_data)[0])
    prediction_clamped = max(0.0, min(100.0, prediction))

    st.subheader("Diagnostic Results")
    
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric(
            label="Estimated Remaining Health",
            value=f"{prediction_clamped:.1f}%"
        )
    
    with res_col2:
        if prediction_clamped >= 85.0:
            st.success("Condition: Excellent. Battery chemistry is in pristine state with minimal capacity fade.")
        elif prediction_clamped >= 75.0:
            st.info("Condition: Good. Normal degradation consistent with steady usage. Consider setting an 80% charge threshold.")
        elif prediction_clamped >= 60.0:
            st.warning("Condition: Moderate Wear. Runtime on battery power will be noticeably reduced compared to factory specs.")
        else:
            st.error("Condition: Advanced Degradation. Severe loss of design capacity. Pack replacement recommended for mobile use.")
