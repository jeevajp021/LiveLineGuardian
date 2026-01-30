import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import os

from pathlib import Path

# Get project root (LiveLineGuardian/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Ensure the plots directory exists
plots_dir = BASE_DIR / 'artifacts' / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)


# 1. Load the model and feature list
with open(BASE_DIR/'artifacts'/'riskanalysis_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    feature_names = data['features']

# 2. Load the training data to provide context to SHAP
# Adjust the path if your file name is different
df = pd.read_csv(BASE_DIR/'data'/'raw'/'safegrid_data.csv')

# 3. Prepare features (same logic as train.py)
X = df.drop(columns=['job_id', 'asset_id', 'crew_id', 'start_time', 'end_time', 'incident_flag', 'severity', 'near_miss_flag'])
X = pd.get_dummies(X, columns=['job_type', 'shift_type'], drop_first=True)
X = X[feature_names].fillna(0)

# 4. Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 5. Generate and Save Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title("Top Risk Drivers in Lineman Safety")
plt.tight_layout()
plt.savefig(plots_dir/'global_risk_factors.png')
print("✅ Global risk factors plot saved to artifacts/plots/global_risk_factors.png")
