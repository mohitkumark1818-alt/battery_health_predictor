# 🔋 Laptop Battery Health Predictor

A Machine Learning powered web application built with **Streamlit** to predict laptop battery health percentage based on usage habits, charging cycles, and operating conditions.

---

## 📌 Project Overview

Laptop battery health degrades over time depending on charging patterns, temperatures, and usage intensity. This project uses a trained machine learning regression model to accurately estimate the remaining battery health percentage given various hardware and user behavioral parameters.

### 🎯 Key Features
- **Interactive UI**: Simple and intuitive web interface built with Streamlit.
- **Instant Predictions**: Get estimated battery health percentage in real-time.
- **Comprehensive Parameters**: Factors in charging frequency, average temperature, heavy load hours, charging level, gaming hours, battery cycles, battery chemistry (Li-ion / Li-Po), fast charging usage, and overall age.

---

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Framework**: [Streamlit](https://streamlit.io/)
- **Data & Modeling**: Scikit-Learn, Pandas, NumPy, Joblib

---

## 📂 Repository Structure
```
├── battery_health_model.pkl            # Trained ML model
├── feature_columns.pkl                 # Feature columns reference
├── scaler.pkl                          # Feature preprocessor / scaler
├── laptop_battery_health_prediction.csv # Dataset
├── laptop_health.py                    # Streamlit application
├── requirements.txt                    # Python dependencies
└── README.md                           # Documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/mohitkumark1818-alt/battery_health_predictor.git
cd battery_health_predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run laptop_health.py
```

After running the command, open your browser and navigate to the local URL (usually `http://localhost:8501`).

---

## 📊 Input Parameters Guide
| Parameter | Description |
|---|---|
| **Charging Frequency Per Day** | How many times the laptop is plugged in per day |
| **Average Temperature (°C)** | Typical operating temperature during use |
| **Heavy Load Hours / Week** | Time spent on intensive tasks (rendering, compilation, etc.) |
| **Average Charge Level (%)** | Typical battery charge maintained |
| **Gaming Hours / Week** | Hours spent gaming weekly |
| **Battery Cycles** | Total charge cycles completed |
| **Battery Type** | Lithium-ion (Li-ion) or Lithium-Polymer (Li-Po) |
| **Fast Charging Usage (%)** | Percentage of time fast charging was used |
| **Average Daily Usage Hours** | Daily uptime in hours |
| **Laptop Age (Months)** | Age of the device in months |
| **Usage Pattern** | Light, Moderate, or Heavy |

---

## 👤 Author
**Mohit Kumawat**
- GitHub: [@mohitkumark1818-alt](https://github.com/mohitkumark1818-alt)
