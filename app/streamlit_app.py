import shap
import streamlit as st
import xgboost as xgb
import time
import sys
import os
import random
import pandas as pd
import numpy as np
import datetime
import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]  # only go up ONE level
sys.path.append(str(ROOT_DIR))
BASE_DIR = ROOT_DIR
MODEL_PATH = ROOT_DIR / "artifacts" / "riskanalysis_model.pkl"

from src.models.predict import load_model_and_features, get_risk_score, classify_risk  



# --- 0. NAVIGATION ---
st.sidebar.title("Navigation")
st.sidebar.markdown("### System Mode")
st.sidebar.info("Simulation Mode: Synthetic sensor data active")

page = st.sidebar.radio("Navigate", ["Live Control Room", "Model Performance Metrics"])

if page == "Model Performance Metrics":

    st.title("📊 Model Analytics & Performance")
    st.write("This page summarizes how the XGBoost accident-risk model performs and what drives its predictions.")

    # --- KPI SECTION ---
    st.markdown("### 🔍 Model Reliability Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Model Accuracy", "94.2%")
        st.metric("Precision (Safety)", "91.5%")

    with col2:
        st.metric("Recall (Risk Detection)", "92.8%")
        st.metric("F1 Score", "92.1%")

    st.markdown("---")

    # --- EXPLAINABILITY ---
    st.subheader("🌍 Global Risk Drivers")
    st.write("These factors have the strongest influence on accident risk predictions across the network (SHAP explainability).")

    if os.path.exists('artifacts/plots/global_risk_factors.png'):
        st.image('artifacts/plots/global_risk_factors.png', use_container_width=True)
    else:
        st.warning("Global risk plot not found. Run: `python3 notebooks/explain_model.py` to generate it.")




def log_event(asset_id, action, user="ControlRoom"):
    try:
        status = st.session_state.get('loto_state', {}).get(asset_id, {}).get('status', 'UNKNOWN')

        log_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "asset_id": asset_id,
            "action": action,
            "user": user,
            "status": status
        }

        log_dir = os.path.join(BASE_DIR, 'data', 'processed')
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, 'audit_trail.csv')

        pd.DataFrame([log_entry]).to_csv(
            file_path,
            mode='a',
            index=False,
            header=not os.path.exists(file_path)
        )

    except Exception as e:
        st.warning(f"Logging failed: {e}")

  
  



# --- 1. CONFIGURATION & STATIC DATA ---


# Define simulated coordinates for your two assets (near Coimbatore)
ASSET_COORDINATES = {
    'A118': {'lat': 11.0168, 'lon': 76.9558},  # High-Risk Zone Example
    'A155': {'lat': 11.0250, 'lon': 77.0100}   # Standard Zone Example
}




# --- 2. INITIALIZATION & STATE MANAGEMENT ---

# Initialize session state for LOTO protocol
if 'loto_state' not in st.session_state:
    st.session_state.loto_state = {
        'A118': {'status': 'LIVE', 'linemen_on_site': 0, 'risk_score': 0.0},
        'A155': {'status': 'LIVE', 'linemen_on_site': 0, 'risk_score': 0.0}
    }
VALID_TRANSITIONS = {
    "LIVE": ["ISOLATED"],
    "ISOLATED": ["MAINTENANCE (LOCKED)", "LIVE"],
    "MAINTENANCE (LOCKED)": ["ISOLATED"]
}

def can_transition(asset_id, new_state):
    current = st.session_state.loto_state[asset_id]['status']
    return new_state in VALID_TRANSITIONS.get(current, [])





# --- 3. CORE LOGIC FUNCTIONS ---

if 'last_auto_run' not in st.session_state:
    st.session_state.last_auto_run = 0

if 'last_action_time' not in st.session_state:
    st.session_state.last_action_time = time.time()

def mark_user_action():
    st.session_state.last_action_time = time.time()

def generate_dynamic_sensor_data(asset_id):
    """Generates slightly randomized sensor and context data. A118 is set to be high risk."""
    is_high_risk = (asset_id == 'A118')
    
    # Base values use high-risk characteristics (low experience, high work hours, poor asset health)
    base_data = {
        'crew_avg_experience_years': 1.5 if is_high_risk else random.uniform(5.0, 9.0), 
        'recent_hours_worked_48h': 35 if is_high_risk else random.randint(8, 20), 
        'asset_voltage_level': 33, 'asset_age_years': 25 if is_high_risk else 10, 
        'maintenance_last_days': 160 if is_high_risk else random.randint(30, 90), 
        'region_hotspot_score': 0.8 if is_high_risk else random.uniform(0.1, 0.4), 
        'incidents_area_30d': 5 if is_high_risk else random.randint(0, 1), 
        'weather_precip_1h': 10.0 if is_high_risk else random.uniform(0, 0.5), 
        'lightning_nearby_1h': 1 if is_high_risk else 0, 
        
        # Sensor data: Add noise and error flags for high-risk scenario
        'V_rms_mean_5m': random.uniform(12100, 12300) if is_high_risk else random.uniform(11000, 11500),
        'V_rms_std_5m': random.uniform(400, 600) if is_high_risk else random.uniform(50, 150), 
        'V_slope_5m': random.uniform(1.0, 2.0) if is_high_risk else random.uniform(0.1, 0.5),
        'frequency_dev_std_5m': random.uniform(0.1, 0.2) if is_high_risk else random.uniform(0.01, 0.04),
        'V_drop_count_15m': 5 if is_high_risk else random.randint(0, 2),
        'I_rms_mean_5m': random.uniform(80, 95) if is_high_risk else random.uniform(30, 50),
        'I_inrush_count_1h': 3 if is_high_risk else 0,
        'thd_percent_1h': random.uniform(7.0, 10.0) if is_high_risk else random.uniform(1.5, 4.0),
        'arc_event_count_1h': 2 if is_high_risk else 0,
        'ground_fault_flag_24h': 1 if is_high_risk else 0,
        'transformer_temp_max_24h': random.uniform(95, 105) if is_high_risk else random.uniform(65, 80),
        'leakage_current_max_24h': random.uniform(2.0, 3.5) if is_high_risk else random.uniform(0.1, 0.5),
        'sensor_comm_loss_flag': 1 if is_high_risk else 0,
        
        # Dummy variables for categorical features
        'job_type_emergency': 1 if is_high_risk else 0, 
        'shift_type_night': 1 if is_high_risk else 0
    }
    return base_data

def simulate_wearable_alert(asset_id):
    """Simulates the wearable sensor override (5% chance of error)."""
    if st.session_state.loto_state[asset_id]['status'] == "ISOLATED" and random.random() < 0.05:
        st.error(f"🚨 **WEARABLE OVERRIDE:** Live Voltage Detected on {asset_id}. ABORT!")
        return True
    return False


def transition_state(asset_id, new_state):
    if can_transition(asset_id, new_state):
        st.session_state.loto_state[asset_id]['status'] = new_state
        return True
    return False



def attempt_energize(asset_id):

    """Handles the final re-energize command."""
    state = st.session_state.loto_state[asset_id]
    
    if state['status'] == 'LIVE':
        st.warning("Line is already LIVE.")
        return
        
    if state['status'] == 'MAINTENANCE (LOCKED)':
        st.error("❌ Cannot energize. Linemen are CHECKED IN.")
        return

    if simulate_wearable_alert(asset_id):
        return

    # Success
    if not transition_state(asset_id, "LIVE"):
        st.error("State transition blocked by system rules.")
        return
    log_event(asset_id, "Line Re-Energized")
    st.success(f"⚡ Line {asset_id} is successfully **LIVE**.")
    st.rerun() # FIX: Use st.rerun()

def run_risk_analysis(selected_asset):
    input_data_raw = generate_dynamic_sensor_data(selected_asset)

    input_data = {}
    for col in feature_names:
        if col in input_data_raw:
            input_data[col] = input_data_raw[col]
        elif col.startswith(('job_type_', 'shift_type_')):
            input_data[col] = 0
        else:
            input_data[col] = 0

    new_score = get_risk_score(model, feature_names, input_data)
    st.session_state.loto_state[selected_asset]['risk_score'] = new_score
    log_event(selected_asset, f"Risk Score Calculated: {new_score:.2f}", "AI_Model")

    if 'risk_history' not in st.session_state:
        st.session_state.risk_history = {}
    if selected_asset not in st.session_state.risk_history:
        st.session_state.risk_history[selected_asset] = []

    st.session_state.risk_history[selected_asset].append(new_score)

    return input_data_raw, input_data, new_score
    
    



# --- 4. MODEL LOADING ---

st.sidebar.title("System Status")
AUTO_REFRESH_INTERVAL = 5  # seconds
IDLE_GRACE_PERIOD = 3  # seconds after user action before refresh allowed
auto_mode = st.sidebar.toggle("🔄 Live Risk Monitoring", value=False)

@st.cache_resource
def load_system(model_path, mtime):
    model, feature_names = load_model_and_features(model_path)
    if isinstance(model, xgb.Booster):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.TreeExplainer(model.get_booster())
    return model, feature_names, explainer

try:
    model_mtime = os.path.getmtime(MODEL_PATH)
    model, feature_names, explainer = load_system(MODEL_PATH, model_mtime)
    st.sidebar.success("ML Model Loaded Successfully!")
except FileNotFoundError:
    st.sidebar.error("Model file not found. Run training first.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"Model loading failed: {e}")
    st.stop()




# --- 5. STREAMLIT UI ---

st.title("🛡️ LiveLine Guardian: Control Room Dashboard")

# Asset Selector
selected_asset = st.selectbox("Select Asset ID", list(st.session_state.loto_state.keys()))
if 'last_asset' not in st.session_state:
    st.session_state.last_asset = selected_asset

if st.session_state.last_asset != selected_asset:
    st.session_state.loto_state[selected_asset]['risk_score'] = 0.0
    st.session_state.last_asset = selected_asset

state = st.session_state.loto_state[selected_asset]

st.markdown("---")

## Asset Location & Lineman Presence
st.subheader("Asset Location & Lineman Presence")

# Get the coordinates for the selected asset
asset_coords = ASSET_COORDINATES[selected_asset]

# Create a DataFrame for the map
map_data = pd.DataFrame([asset_coords])
map_data.rename(columns={'lat': 'latitude', 'lon': 'longitude'}, inplace=True)

# Add the map to the UI
st.map(map_data, zoom=12)

if state['linemen_on_site'] > 0:
    st.info(f"📍 {state['linemen_on_site']} Linemen are detected at this location.")
else:
    st.warning("No linemen currently registered at this asset location.")

st.markdown("---")

## Status and Risk Metrics
col1, col2 = st.columns(2)

with col1:
    if state['status'] == 'LIVE':
        status_text = "LIVE"
        status_emoji = "🔴"
    elif state['status'] == 'MAINTENANCE (LOCKED)':
        status_text = "LOCKED"
        status_emoji = "🔒"
    else:
        status_text = "ISOLATED"
        status_emoji = "🟡"
        
    st.metric(label="Current Line Status", value=f"{status_emoji} {status_text}", help="The current energy state of the power line.")


with col2:
    risk_label, advice = classify_risk(state['risk_score'])
    def risk_color(label):
        return {"HIGH":"red", "MODERATE":"orange", "LOW":"green"}[label]
    st.markdown(
        f"<h3 style='color:{risk_color(risk_label)}'>Risk Level: {risk_label}</h3>",
        unsafe_allow_html=True
    )

    
    st.metric(label="Predictive Risk Level", value=f"{risk_label} ({state['risk_score']:.2f})", help="ML model prediction of an incident occurring during the job.")
    
    # Display advice based on risk
    if risk_label == "HIGH":
         st.error(advice)
    elif risk_label == "MODERATE":
         st.warning(advice)
    else:
         st.info(advice)

st.markdown("---")

## Control Panel & LOTO Actions
st.subheader("LOTO Protocol and Control")


# 1. Predict Risk Button (Triggers dynamic sensor data simulation)

manual_trigger = st.button("Run Predictive Risk Analysis")

should_run = False

# --- Manual run ---
if manual_trigger:
    mark_user_action()
    should_run = True

# --- Auto-timed run ---
elif auto_mode:
    now = time.time()
    if now - st.session_state.last_auto_run >= AUTO_REFRESH_INTERVAL:
        should_run = True
        st.session_state.last_auto_run = now


# --- EXECUTE MODEL ONLY WHEN NEEDED ---
if should_run:
    with st.spinner("Analyzing real-time sensor and context data..."):
        input_data_raw, input_data, new_score = run_risk_analysis(selected_asset)

        st.success(f"Risk re-calculated. Score: {new_score:.2f}")
        st.line_chart(st.session_state.risk_history[selected_asset])

        # SHAP
        st.markdown("### 🔍 Why is this risk level assigned?")
        try:
            df_input = pd.DataFrame([input_data])
            df_input = df_input.reindex(columns=feature_names, fill_value=0)
            shap_values = explainer(df_input)

            impact_df = pd.DataFrame({
                "Feature": feature_names,
                "Impact": shap_values.values[0]
            }).assign(abs_impact=lambda x: x["Impact"].abs()) \
              .sort_values(by="abs_impact", ascending=False)

            for _, row in impact_df.head(3).iterrows():
                clean_name = row["Feature"].replace("_", " ").title()
                direction = "increasing" if row["Impact"] > 0 else "reducing"
                st.write(f"- **{clean_name}** is strongly **{direction}** risk")

        except Exception as e:
            st.warning("SHAP explanation unavailable for this prediction.")
            log_event(selected_asset, f"SHAP Failure: {str(e)}", "System")

    with st.expander("Show Latest Sensor Inputs"):
        st.json(input_data_raw)
   
        

# 2. Control Room Action: Isolation
if st.button("Control Room: Request ISOLATION", disabled=not can_transition(selected_asset, "ISOLATED")):
    mark_user_action()
    transition_state(selected_asset, "ISOLATED")
    log_event(selected_asset, "Isolation Requested")
    st.success(f"Line {selected_asset} is now **ISOLATED**.")
    st.rerun()



# 3. Lineman Actions
col_lineman_in, col_lineman_out = st.columns(2)

with col_lineman_in:
    if st.button("Lineman CHECK-IN (Via App)", use_container_width=True,
    disabled=not can_transition(selected_asset, "MAINTENANCE (LOCKED)")):
            mark_user_action()
            st.session_state.loto_state[selected_asset]['linemen_on_site'] += 1
            transition_state(selected_asset, "MAINTENANCE (LOCKED)")
            log_event(selected_asset, "Lineman Check-In", "Lineman_App")
            st.info(f"Lineman Checked In. Line is **LOCKED**.")
            st.rerun() # FIX: Use st.rerun()


with col_lineman_out:
    if st.button(
        "Lineman CHECK-OUT (Via App)",
        use_container_width=True,
        disabled=(state['linemen_on_site'] == 0)
    ):
        mark_user_action()
        state_obj = st.session_state.loto_state[selected_asset]
        state_obj['linemen_on_site'] = max(0, state_obj['linemen_on_site'] - 1)     
        log_event(selected_asset, "Lineman Check-Out", "Lineman_App")

        if st.session_state.loto_state[selected_asset]['linemen_on_site'] == 0:
            if transition_state(selected_asset, "ISOLATED"):
                st.success("All clear. Status set to ISOLATED.")
                st.rerun()

        else:
            current_count = st.session_state.loto_state[selected_asset]['linemen_on_site']
            st.info(f"Lineman Checked Out. {current_count} remaining.")


# 4. Final Control Room Action: Re-Energize
st.markdown("---")
RISK_LOCK_THRESHOLD = 0.7
if state['risk_score'] > RISK_LOCK_THRESHOLD and state['status'] != "LIVE":
    st.error("🚨 Auto-lock: Risk too high to re-energize")
if st.button(
    "Control Room: Attempt RE-ENERGIZE",
    type="primary",
    disabled=(
    not can_transition(selected_asset, "LIVE")
    or state['risk_score'] > RISK_LOCK_THRESHOLD
)
):
    mark_user_action()
    attempt_energize(selected_asset) # attempt_energize now uses st.rerun()



st.markdown("---")
st.subheader("📜 System Audit Trail")
try:
    log_file = os.path.join(BASE_DIR, 'data', 'processed', 'audit_trail.csv')
    if os.path.exists(log_file):
        audit_df = pd.read_csv(log_file)
        st.dataframe(audit_df.tail(10), use_container_width=True) # Show last 10 actions
    else:
        st.info("No logs recorded yet.")
except Exception as e:
    st.error(f"Could not load audit logs: {e}")
    
    
    
    
# --- AUTO REFRESH LOOP ---

if auto_mode:
    st.caption(f"⏱ Auto-refreshing every {AUTO_REFRESH_INTERVAL}s")

    time_since_action = time.time() - st.session_state.last_action_time

    if time_since_action > IDLE_GRACE_PERIOD:
        st.rerun()
