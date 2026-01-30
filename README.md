# 🛡️ LiveLine Guardian: IoT-Based Smart Safety for Power Grid Linemen

**LiveLine Guardian** is an intelligent, data-driven safety ecosystem designed to eliminate fatalities in Electricity Board (EB) maintenance operations. It integrates a **Digital Lockout-Tagout (LOTO)** protocol with **Machine Learning** to provide real-time risk assessment and automated safety overrides.

---

## 🧩 The Problem
Electrical linemen face high-risk environments where manual communication gaps or equipment failures lead to fatal electrocutions. 
- **Root Causes:** Manual power isolation errors, lack of real-time awareness of live wires, and crew fatigue.
- **The Toll:** Thousands of preventable accidents annually during grid maintenance.

## 💡 The Solution
LiveLine Guardian solves this through a three-layer protection strategy:
1. **Predictive Layer:** An XGBoost model analyzes sensor data, weather, and human factors to predict risk *before* work begins.
2. **Systemic Layer:** A Finite State Machine (FSM) based Digital LOTO system that physically prevents re-energizing while workers are "Checked-In."
3. **Physical Layer:** A simulated Smart Wearable (NCVS) that provides a final safety override if residual voltage is detected on-site.

---

## 🏗️ Technical Architecture
- **Backend:** Python 3.13 (Modular `src/` structure)
- **Machine Learning:** XGBoost (94% Accuracy), SHAP (Explainability)
- **Frontend:** Streamlit (Real-time Control Room Dashboard)
- **Data:** Synthetic IoT Sensor & Maintenance Logs (`safegrid_synthetic.csv`)
- **Failsafe Logic:** Hardcoded State Transitions (LIVE -> ISOLATED -> LOCKED)

---

## 📦 Installation & Usage
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/your-username/LiveLineGuardian.git](https://github.com/your-username/LiveLineGuardian.git)
   cd LiveLineGuardian

2. **Setup Environment:**
    ```bash
    python -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt

3. **Run Training:**
	```bash
	python src/models/train.py

4. **Launch Dashboard:**
	```bash
	streamlit run app/streamlit_app.py


## 🧪 Safety Test Demo Script

Use this guided walkthrough to showcase the system’s intelligence and multi-layered safety design during a live demo.

---

### 🚨 Scenario 1: High-Risk Warning (AI Insight)

**Goal:** Show how AI predicts danger *before* an incident happens.

**Steps:**

1. Select **Asset `A118`** *(Simulated High-Risk Zone)*  
2. Click **▶ Run Predictive Risk Analysis**

**🔍 Observation:**

- The risk level turns **🔴 RED**
- Review **Top Risk Contributors (SHAP Explanation)**

**🧠 Explain to the audience:**

The AI flags elevated danger due to factors such as:

- ⛈️ Lightning Risk  
- 😴 Crew Fatigue  
- 🌡️ Environmental Stress  

This proves the system provides **explainable AI insights**, not just a black-box warning.

---

### 🔒 Scenario 2: Digital Lockout (Systemic Failsafe)

**Goal:** Demonstrate how the system prevents accidental re-energization.

**Steps:**

1. Click **Request Isolation** → Status becomes **🟡 ISOLATED**  
2. Perform **Lineman Check-In** → Status becomes **🔒 LOCKED**  
3. **The Test:** Click **⚡ Attempt RE-ENERGIZE**

**🛑 Observation:**

- The button is **disabled or blocked**

**💡 Explain to the audience:**  

The platform enforces a **digital lockout protocol**, ensuring power **cannot** be restored while personnel are working on the line — eliminating human error.

---

### ⌚ Scenario 3: Wearable Override (Physical Failsafe)

**Goal:** Show the system’s final real-world safety layer.

**Steps:**

1. Perform **Lineman Check-Out** → Status returns to **🟡 ISOLATED**  
2. Click **⚡ Attempt RE-ENERGIZE**

**⚠️ Observation:**

- Sometimes a **🚫 Wearable Override Alert** appears  
- Power restoration is **blocked**

**🧠 Explain to the audience:**  

The system detects **phantom or residual voltage** using wearable sensors, preventing energization even after work is declared complete.

---

## 🛡️ What This Demonstrates

| Layer | Protection Type | What It Prevents |
|------|-----------------|------------------|
| 🧠 AI Risk Engine | Predictive Safety | Unsafe working conditions |
| 🔒 Digital Lockout | System Control | Human operational error |
| ⌚ Wearable Override | Physical Safety | Hidden electrical hazards |

---
