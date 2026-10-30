# Used Car Price Prediction

This project turns the classroom used-car notebook into a safer deployable
workflow. It saves **one fitted scikit-learn Pipeline**, not a separate model
and preprocessor. The Streamlit app calls `pipeline.predict(input_dataframe)`,
which guarantees the saved preprocessor runs before XGBoost.

## Files

- `usedcarpricepredictor.py` — the supervised class notebook export, updated
  to save one combined deployment artifact at the end.
- `train_used_car_model.py` — repeatable production training script.
- `used_car_app.py` — Streamlit interface.
- `used_car_price_pipeline.joblib` — created after training; this is the only
  model file the app needs.

## Run locally

Create a virtual environment, install the dependencies, then train from the
Kaggle `used_cars.csv` used in class:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_used_car_model.py "C:\path\to\used_cars.csv"
streamlit run used_car_app.py
```

The model is trained on `log1p(price)`. The pipeline returns this log value;
the app performs `expm1` once, immediately before displaying the dollar price.
