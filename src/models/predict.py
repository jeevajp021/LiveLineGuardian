# src/models/predict.py
import pickle
import pandas as pd
import xgboost as xgb
from pathlib import Path


def load_model_and_features(artifacts_path):
    """
    Loads XGBoost JSON model + feature list.
    artifacts_path should point to the artifacts folder.
    """

    artifacts_path = Path(artifacts_path)

    # Load model (stable format)
    model = xgb.Booster()
    model.load_model(str(artifacts_path / "riskanalysis_model.json"))

    # Load feature list
    with open(artifacts_path / "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    return model, feature_names


def get_risk_score(model, feature_cols, new_job_data):
    """
    Predicts incident probability using XGBoost Booster
    """

    df_new = pd.DataFrame([new_job_data])

    # Ensure all expected columns exist
    for col in feature_cols:
        if col not in df_new.columns:
            df_new[col] = 0

    # Correct column order
    df_new = df_new[feature_cols]

    # Convert to DMatrix (required for Booster)
    dmatrix = xgb.DMatrix(df_new)

    # Booster returns probability directly for binary:logistic
    prob = float(model.predict(dmatrix)[0])
    return prob


def classify_risk(probability):
    """Converts probability to a safety-centric risk level."""
    if probability >= 0.7:
        return "HIGH", "🚨 Immediate attention required, consider cancelling job."
    elif probability >= 0.3:
        return "MODERATE", "⚠️ Proceed with caution, mandatory senior supervision."
    else:
        return "LOW", "✅ Standard safety procedures apply."
