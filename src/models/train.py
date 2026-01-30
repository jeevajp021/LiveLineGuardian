# src/models/train.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def train_safegrid_model(data_path, model_path):
    print("Starting robust model training...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(data_path) 
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {data_path}. Please check the file name and path.")
        return None, None

    # 2. Select Features and Target
    features = df.drop(columns=['job_id', 'asset_id', 'crew_id', 'start_time', 'end_time', 'incident_flag', 'severity', 'near_miss_flag'])
    target = df['incident_flag']

    # 3. Preprocessing: Convert categorical variables to one-hot encoding
    # We explicitly only convert the two known categorical columns
    features = pd.get_dummies(features, columns=['job_type', 'shift_type'], drop_first=True)
    
    # 4. ROBUSTNESS STEP: Ensure all features passed to XGBoost are numeric
    # This automatically drops any column that pandas could not convert to a number,
    # preventing runtime errors if data quality is inconsistent.
    numeric_features = features.apply(pd.to_numeric, errors='coerce')
    
    # 5. Handle Missing Data (Imputation)
    # Fill NaN values (created by 'coerce' or originally in the data) with 0.
    numeric_features = numeric_features.fillna(0) 

    # 6. Train Model
    X_train, _, y_train, _ = train_test_split(numeric_features, target, test_size=0.2, random_state=42)
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    # 7. Save Model and Feature Names
    with open(model_path, 'wb') as file:
        pickle.dump({'model': model, 'features': list(numeric_features.columns)}, file)

    print(f"✅ Robust ML Model trained and saved to {model_path}.")
    return model, list(numeric_features.columns)

if __name__ == '__main__':
    # Ensure this path is correct relative to where you run the script (project root)
    TRAIN_DATA_PATH = 'data/raw/safegrid_data.csv'
    MODEL_SAVE_PATH = 'artifacts/riskanalysis_model.pkl'
    train_safegrid_model(TRAIN_DATA_PATH, MODEL_SAVE_PATH)
