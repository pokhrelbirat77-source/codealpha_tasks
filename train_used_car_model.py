"""Train and save the used-car price estimator as one deployable Pipeline.

The classroom notebook is retained in ``usedcarpricepredictor.py``.  This
script is the repeatable training entry point used by the Streamlit app.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


FEATURE_COLUMNS = [
    "brand",
    "model_year",
    "milage",  # This spelling matches the source dataset.
    "fuel_type",
    "transmission_type",
    "accident",
    "clean_title",
    "horsepower",
    "engine_liters",
    "cylinders",
]

NUMERIC_COLUMNS = [
    "model_year",
    "milage",
    "horsepower",
    "engine_liters",
    "cylinders",
    "accident",
    "clean_title",
]

CATEGORICAL_COLUMNS = ["brand", "fuel_type", "transmission_type"]


def simplify_transmission(value: object) -> str:
    """Apply the exact three-class transmission rule from the class notebook."""
    text = str(value).lower().strip()
    if "manual" in text or "m/t" in text or "mt" in text:
        return "Manual"
    if any(
        term in text
        for term in (
            "automatic",
            "a/t",
            "at",
            "cvt",
            "dual shift",
            "cmdshft",
            "overdrive",
            "single-speed fixed gear",
        )
    ):
        return "Automatic"
    return "Other"


def extract_horsepower(value: object) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)HP", str(value))
    return float(match.group(1)) if match else np.nan


def extract_engine_liters(value: object) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)L", str(value))
    return float(match.group(1)) if match else np.nan


def extract_cylinders(value: object) -> float:
    text = str(value).lower()
    for pattern in (r"(\d+)\s*cylinder", r"\bv(\d+)\b", r"\bi(\d+)\b"):
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return np.nan


def clean_training_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the feature engineering used in the supervised class project."""
    missing_columns = {
        "brand", "model_year", "milage", "fuel_type", "transmission",
        "accident", "clean_title", "engine", "price",
    }.difference(raw_data.columns)
    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    clean_data = raw_data.copy()
    clean_data["milage"] = pd.to_numeric(
        clean_data["milage"].astype(str).str.replace(" mi.", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce",
    )
    clean_data["price"] = pd.to_numeric(
        clean_data["price"].astype(str).str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce",
    )
    clean_data["model_year"] = pd.to_numeric(clean_data["model_year"], errors="coerce")

    clean_data["fuel_type"] = clean_data["fuel_type"].replace(
        ["–", "not supported"], np.nan
    ).fillna("Unknown")
    clean_data["accident"] = clean_data["accident"].map(
        {
            "None reported": 0,
            "At least 1 accident or damage reported": 1,
        }
    ).fillna(-1)
    clean_data["clean_title"] = clean_data["clean_title"].map({"Yes": 1}).fillna(0)
    clean_data["transmission_type"] = clean_data["transmission"].apply(
        simplify_transmission
    )
    clean_data["horsepower"] = clean_data["engine"].apply(extract_horsepower)
    clean_data["engine_liters"] = clean_data["engine"].apply(extract_engine_liters)
    clean_data["cylinders"] = clean_data["engine"].apply(extract_cylinders)

    # The same target range used by the final notebook model.
    return clean_data.loc[
        clean_data["price"].between(500, 300_000), FEATURE_COLUMNS + ["price"]
    ].copy()


def build_pipeline() -> Pipeline:
    """Build one object that owns feature preprocessing and the regressor."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42,
                    objective="reg:squarederror",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train(csv_path: Path, output_path: Path) -> dict[str, float | int]:
    data = clean_training_data(pd.read_csv(csv_path))
    if len(data) < 10:
        raise ValueError("At least 10 valid training rows are required.")

    features = data.loc[:, FEATURE_COLUMNS]
    log_price = np.log1p(data["price"])
    X_train, X_test, y_train, y_test = train_test_split(
        features, log_price, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predicted_price = np.expm1(pipeline.predict(X_test))
    actual_price = np.expm1(y_test)
    metrics: dict[str, float | int] = {
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae_usd": float(mean_absolute_error(actual_price, predicted_price)),
        "rmse_usd": float(mean_squared_error(actual_price, predicted_price) ** 0.5),
        "r2": float(r2_score(actual_price, predicted_price)),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    output_path.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the complete used-car price prediction pipeline."
    )
    parser.add_argument("csv_path", type=Path, help="Path to used_cars.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("used_car_price_pipeline.joblib"),
        help="Destination for the single saved deployment pipeline.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = train(args.csv_path, args.output)
    print(f"Saved complete pipeline to: {args.output.resolve()}")
    print(f"Test MAE:  ${results['mae_usd']:,.2f}")
    print(f"Test RMSE: ${results['rmse_usd']:,.2f}")
    print(f"Test R²:   {results['r2']:.4f}")
