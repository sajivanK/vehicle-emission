import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# =========================
# Load trained model & pipeline
# =========================
model = joblib.load("linear_regression_model.pkl")
pipeline = joblib.load("preprocessing_pipeline.pkl")

# Load your dataset (for average comparisons)
df = pd.read_csv("cleaned_vehicle_dataset.csv")  # <- replace with your cleaned dataset path

# =========================
# App Layout
# =========================
st.set_page_config(page_title="CO₂ Emissions Predictor", page_icon="🚗", layout="wide")

st.title("🚗 Vehicle CO₂ Emissions Predictor")
st.markdown("Predict vehicle **CO₂ emissions (g/km)** based on specifications using a Multiple Linear Regression model.")

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("🔧 Enter Vehicle Specifications")

engine_size = st.sidebar.number_input("Engine size (L)", min_value=0.5, max_value=10.0, step=0.1, value=2.0)
cylinders = st.sidebar.number_input("Number of Cylinders", min_value=3, max_value=12, step=1, value=4)
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "Ethanol", "Natural Gas", "Premium Petrol"])
combined_l_100km = st.sidebar.number_input("Combined (L/100 km)", min_value=2.0, max_value=25.0, step=0.1, value=8.5)

# Predict button
if st.sidebar.button("Predict"):
    # Convert input to DataFrame
    input_data = pd.DataFrame({
        "Engine size (L)": [engine_size],
        "Cylinders": [cylinders],
        "Fuel type": [fuel_type],
        "Combined (L/100 km)": [combined_l_100km]
    })

    # Preprocess & predict
    input_processed = pipeline.transform(input_data)
    prediction = model.predict(input_processed)[0]

    # =========================
    # Results Section
    # =========================
    st.subheader("🔮 Predicted CO₂ Emissions")
    st.metric("CO₂ emissions (g/km)", f"{prediction:.2f}")

    # Dataset average comparison
    avg_emission = df["CO2 emissions (g/km)"].mean()

    # Model evaluation (on dataset, for display)
    X = df.drop(columns=["CO2 emissions (g/km)", "CO2 rating", "Smog rating"], errors="ignore")
    y = df["CO2 emissions (g/km)"]
    X_processed = pipeline.transform(X)
    y_pred = model.predict(X_processed)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    st.write("---")
    st.subheader("📊 Model Performance")
    st.write(f"**R² Score:** {r2:.4f}")
    st.write(f"**RMSE:** {rmse:.2f} g/km")
    st.write(f"**MAE:** {mae:.2f} g/km")

    # =========================
    # Graphs Section
    # =========================
    col1, col2 = st.columns(2)

    # 1. Actual vs Predicted Scatter Plot
    with col1:
        st.write("### 📈 Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(5,4))
        sns.scatterplot(x=y, y=y_pred, alpha=0.6, ax=ax)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
        ax.set_xlabel("Actual CO₂ emissions (g/km)")
        ax.set_ylabel("Predicted CO₂ emissions (g/km)")
        ax.set_title(f"R² = {r2:.3f}")
        st.pyplot(fig)

    # 2. Residuals Plot
    with col2:
        st.write("### 📊 Residuals Distribution")
        residuals = y - y_pred
        fig, ax = plt.subplots(figsize=(5,4))
        sns.histplot(residuals, bins=30, kde=True, ax=ax, color="purple")
        ax.set_xlabel("Prediction Error (g/km)")
        ax.set_title("Residuals Distribution")
        st.pyplot(fig)

    # 3. Feature Coefficient Impact
    st.write("---")
    st.write("### ⚡ Feature Impact on CO₂ Emissions")
    encoded_features = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coef_df = pd.DataFrame({
        "Feature": encoded_features,
        "Coefficient (β)": model.coef_
    }).sort_values(by="Coefficient (β)", ascending=False)

    fig, ax = plt.subplots(figsize=(7,5))
    sns.barplot(x="Coefficient (β)", y="Feature", data=coef_df, palette="coolwarm", ax=ax)
    ax.set_title("Regression Coefficients")
    st.pyplot(fig)

    # 4. User vs Dataset Average Comparison
    st.write("---")
    st.write("### 🚘 Your Vehicle vs Dataset Average")
    comparison_df = pd.DataFrame({
        "Category": ["Your Vehicle", "Dataset Average"],
        "CO₂ emissions (g/km)": [prediction, avg_emission]
    })
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x="Category", y="CO₂ emissions (g/km)", data=comparison_df, palette="Set2", ax=ax)
    st.pyplot(fig)

# =========================
# Footer
# =========================
st.write("---")
st.markdown("Built with ❤️ using **Python, Streamlit, and Scikit-learn**  \nProject by: *Sajivan & Team*")
