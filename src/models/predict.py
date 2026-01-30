# src/models/predict.py
import pickle
import pandas as pd

def load_model_and_features(model_path):
    """Loads the trained model and the feature list it requires."""
    with open(model_path, 'rb') as file:
        data = pickle.load(file)
    return data['model'], data['features']

def get_risk_score(model, feature_cols, new_job_data):
    """Predicts the risk probability for a new job/sensor input."""
    df_new = pd.DataFrame([new_job_data])

    # Ensure input data matches the model's feature structure
    for col in feature_cols:
        if col not in df_new.columns:
            df_new[col] = 0

    # Enforce correct column order
    df_new = df_new[feature_cols]

    # Predict probability of incident (1)
    prob = model.predict_proba(df_new)[0, 1]
    return prob

def classify_risk(probability):
    """Converts probability to a safety-centric risk level."""
    if probability >= 0.7:
        return "HIGH", "🚨 Immediate attention required, consider cancelling job."
    elif probability >= 0.3:
        return "MODERATE", "⚠️ Proceed with caution, mandatory senior supervision."
    else:
        return "LOW", "✅ Standard safety procedures apply."
