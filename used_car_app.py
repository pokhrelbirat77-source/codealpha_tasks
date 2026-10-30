"""Streamlit UI for the one-file used-car price prediction pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("used_car_price_pipeline.joblib")
FEATURE_COLUMNS = [
    "brand",
    "model_year",
    "milage",
    "fuel_type",
    "transmission_type",
    "accident",
    "clean_title",
    "horsepower",
    "engine_liters",
    "cylinders",
]


@st.cache_resource
def load_pipeline(model_path: str):
    """Load and validate the one saved Pipeline artifact once per server."""
    pipeline = joblib.load(model_path)
    if not hasattr(pipeline, "named_steps"):
        raise TypeError("The saved file is not a scikit-learn Pipeline.")
    if {"preprocessor", "model"}.difference(pipeline.named_steps):
        raise ValueError(
            "The saved pipeline must contain 'preprocessor' and 'model' steps."
        )

    trained_columns = list(getattr(pipeline, "feature_names_in_", []))
    if trained_columns and trained_columns != FEATURE_COLUMNS:
        raise ValueError(
            "The saved pipeline was trained with a different input schema: "
            + ", ".join(trained_columns)
        )
    return pipeline


def options_from_pipeline(pipeline, column: str, fallback: list[str]) -> list[str]:
    """Show known categories while still allowing the pipeline to handle unknowns."""
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        transformer = preprocessor.named_transformers_["cat"]
        category_columns = list(preprocessor.transformers_[1][2])
        index = category_columns.index(column)
        return [str(value) for value in transformer.categories_[index]]
    except (AttributeError, IndexError, KeyError, ValueError):
        return fallback


st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Used Car Price Predictor")
st.caption(
    "Educational estimate based on the class XGBoost model. The saved pipeline "
    "performs the same imputation, scaling, and encoding used during training."
)

if not MODEL_PATH.exists():
    st.error("The trained pipeline file is not available yet.")
    st.code(
        "python train_used_car_model.py path\\to\\used_cars.csv "
        "--output used_car_price_pipeline.joblib",
        language="powershell",
    )
    st.stop()

try:
    model_pipeline = load_pipeline(str(MODEL_PATH))
except Exception as error:
    st.error(f"Could not load the used-car model pipeline: {error}")
    st.stop()

brands = options_from_pipeline(model_pipeline, "brand", ["Ford", "Toyota", "Honda"])
fuel_types = options_from_pipeline(
    model_pipeline, "fuel_type", ["Gasoline", "Hybrid", "Electric", "Diesel", "Unknown"]
)
transmissions = options_from_pipeline(
    model_pipeline, "transmission_type", ["Automatic", "Manual", "Other"]
)

form_column, result_column = st.columns([1.2, 1], gap="large")

with form_column:
    st.subheader("Vehicle details")
    with st.form("car_details"):
        brand = st.selectbox("Brand", brands)
        model_year = st.number_input(
            "Model year", min_value=1980, max_value=2027, value=2020, step=1
        )
        milage = st.number_input(
            "Mileage (miles)", min_value=0, max_value=1_500_000, value=50_000, step=1_000
        )
        fuel_type = st.selectbox("Fuel type", fuel_types)
        transmission_type = st.selectbox("Transmission", transmissions)
        horsepower = st.number_input(
            "Horsepower", min_value=0.0, max_value=2_000.0, value=180.0, step=5.0
        )
        engine_liters = st.number_input(
            "Engine size (liters)", min_value=0.0, max_value=12.0, value=2.0, step=0.1
        )
        cylinders = st.number_input(
            "Cylinders", min_value=0, max_value=16, value=4, step=1
        )
        accident_label = st.radio(
            "Reported accident or damage?", ["No", "Yes", "Unknown"], horizontal=True
        )
        clean_title_label = st.radio("Clean title?", ["Yes", "No"], horizontal=True)
        submitted = st.form_submit_button("Estimate price", type="primary")

with result_column:
    st.subheader("Estimated market price")
    if not submitted:
        st.info("Enter the vehicle details, then select **Estimate price**.")
    else:
        accident = {"No": 0, "Yes": 1, "Unknown": -1}[accident_label]
        clean_title = 1 if clean_title_label == "Yes" else 0
        input_data = pd.DataFrame(
            [[
                brand,
                float(model_year),
                float(milage),
                fuel_type,
                transmission_type,
                float(accident),
                float(clean_title),
                float(horsepower),
                float(engine_liters),
                float(cylinders),
            ]],
            columns=FEATURE_COLUMNS,
        )

        try:
            # Calling predict on the Pipeline always runs the fitted
            # preprocessor first, then the fitted XGBoost regressor.
            predicted_log_price = float(model_pipeline.predict(input_data)[0])
            predicted_price = float(np.expm1(predicted_log_price))
            if not np.isfinite(predicted_price) or predicted_price < 0:
                raise ValueError("The model returned an invalid price.")

            st.metric("Predicted price", f"${predicted_price:,.0f} USD")
            st.success("Prediction generated using the saved training pipeline.")
            with st.expander("Model input sent to the pipeline"):
                st.dataframe(input_data, hide_index=True, width="stretch")
        except Exception as error:
            st.error(f"Prediction could not be generated: {error}")

st.divider()
st.caption(
    "Predictions are educational estimates, not valuations. The model was trained "
    "on listings priced from $500 to $300,000, so results outside that range need care."
)
