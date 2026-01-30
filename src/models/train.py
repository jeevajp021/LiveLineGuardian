# src/models/train.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from pathlib import Path

def train_safegrid_model(data_path):
    print("Starting robust model training...")

    BASE_DIR = Path(__file__).resolve().parents[2]
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # 1. Load Data
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {data_path}.")
        return None, None

    # 2. Select Features and Target
    features = df.drop(columns=[
        'job_id', 'asset_id', 'crew_id',
        'start_time', 'end_time',
        'incident_flag', 'severity', 'near_miss_flag'
    ])
    target = df['incident_flag']

    # 3. One-hot encode
    features = pd.get_dummies(features, columns=['job_type', 'shift_type'], drop_first=True)

    # 4. Ensure numeric
    numeric_features = features.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 5. Train
    X_train, _, y_train, _ = train_test_split(numeric_features, target, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    feature_names = list(numeric_features.columns)

    # 🔥 6. SAVE IN PRODUCTION FORMAT

    # Save stable XGBoost model
    booster = model.get_booster()
    booster.save_model(ARTIFACTS_DIR / "riskanalysis_model.json")

    # Save feature list
    with open(ARTIFACTS_DIR / "feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    print("✅ Model saved as JSON + feature_names.pkl")
    return booster, feature_names


if __name__ == '__main__':
    TRAIN_DATA_PATH = 'data/raw/safegrid_data.csv'
    train_safegrid_model(TRAIN_DATA_PATH)
