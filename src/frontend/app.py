"""Streamlit Retention Analytics Dashboard for Customer Churn Prediction.

Consumes the Phase 7 FastAPI serving layer via HTTP JSON requests.
Does NOT perform direct model training or preprocessing in Streamlit.
"""

import os
from typing import Any, Dict, Optional, Tuple
import requests
import streamlit as st

# Configurable API Endpoint
API_BASE_URL = os.getenv("CHURN_API_URL", "http://127.0.0.1:8000").rstrip("/")

# Preset Customer Profiles for Demonstration
PRESET_PROFILES = {
    "Custom Input": None,
    "Preset: Low-Risk Loyal Customer": {
        "customerID": "CUST-LOW-001",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65.0,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 60.00,
        "TotalCharges": 3900.00,
    },
    "Preset: Moderate-Risk Intermediate Customer": {
        "customerID": "CUST-MED-002",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 24.0,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 80.00,
        "TotalCharges": 1920.00,
    },
    "Preset: High-Risk Attrition Customer": {
        "customerID": "CUST-HIGH-003",
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2.0,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.00,
        "TotalCharges": 190.00,
    },
}

RETENTION_GUIDANCE = {
    "Low Risk": (
        "**Recommended Action:** Continue standard engagement and monitor account health. "
        "No special retention discounts or manual intervention required at this time."
    ),
    "Medium Risk": (
        "**Recommended Action:** Consider proactive engagement, onboarding satisfaction check-in, "
        "or self-service feature recommendations to deepen product adoption."
    ),
    "High Risk": (
        "**Recommended Action:** Prioritize for proactive retention outreach. Review service ticket history, "
        "investigate possible service friction, and offer a 1-year contract incentive."
    ),
    "Critical Risk": (
        "**Recommended Action:** Prioritize for immediate executive/account manager retention review. "
        "Deploy targeted contract renewal credits or customized service bundles to mitigate imminent attrition."
    ),
}


def check_api_health(base_url: str = API_BASE_URL) -> bool:
    """Check if FastAPI backend is online."""
    try:
        resp = requests.get(f"{base_url}/health", timeout=3)
        return resp.status_code == 200 and resp.json().get("status") == "healthy"
    except Exception:
        return False


def get_model_info(base_url: str = API_BASE_URL) -> Optional[Dict[str, Any]]:
    """Fetch model metadata from FastAPI backend."""
    try:
        resp = requests.get(f"{base_url}/model-info", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def request_prediction(payload: Dict[str, Any], base_url: str = API_BASE_URL) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Send prediction request to FastAPI /predict-churn endpoint."""
    url = f"{base_url}/predict-churn"
    try:
        resp = requests.post(url, json=payload, timeout=5)
        if resp.status_code == 200:
            return resp.json(), None
        elif resp.status_code == 422:
            detail = resp.json().get("detail", "Input validation error.")
            return None, f"Input Validation Error (HTTP 422): {detail}"
        else:
            return None, f"API Error (HTTP {resp.status_code}): {resp.text}"
    except requests.exceptions.ConnectionError:
        return None, f"Cannot connect to FastAPI backend at {base_url}. Please ensure Uvicorn is running."
    except requests.exceptions.Timeout:
        return None, "API request timed out. Please verify backend responsiveness."
    except Exception as e:
        return None, f"Unexpected error during API communication: {str(e)}"


def render_dashboard() -> None:
    """Main Streamlit application layout."""
    st.set_page_config(
        page_title="Customer Churn Prediction & Retention Analytics",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header / Overview Section
    st.title("📊 Customer Churn Prediction & Retention Analytics")
    st.markdown(
        """
        This operational analytics dashboard predicts individual customer churn probabilities and segments accounts into 
        **Operational Churn Risk Bands** to support targeted retention workflows.
        
        *All predictions and feature transformations are processed in real-time through the **FastAPI model-serving backend**.*
        """
    )

    # -------------------------------------------------------------
    # Sidebar: Backend Connection & Demo Presets
    # -------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ System Status")
        is_healthy = check_api_health(API_BASE_URL)
        if is_healthy:
            st.success("🟢 API Status: Connected", icon="✅")
            model_info = get_model_info(API_BASE_URL)
            if model_info:
                st.caption(f"**Model:** {model_info.get('model', 'Logistic Regression')}")
                st.caption(f"**Features Ingested:** {model_info.get('feature_count', 47)}")
                st.caption(f"**Decision Threshold:** {model_info.get('decision_threshold', 0.50):.2f}")
        else:
            st.error("🔴 API Status: Unavailable", icon="⚠️")
            st.warning(
                f"FastAPI backend is unreachable at `{API_BASE_URL}`.\n\n"
                "Please start the backend server:\n"
                "```powershell\nuvicorn src.api.main:app --reload\n```"
            )

        st.divider()
        st.header("🧪 Demo Customer Presets")
        selected_preset = st.selectbox(
            "Load Sample Customer Profile:",
            options=list(PRESET_PROFILES.keys()),
            index=0,
            help="Select a realistic customer profile to automatically populate input fields.",
        )
        preset_data = PRESET_PROFILES[selected_preset] or {}

    # -------------------------------------------------------------
    # Customer Profile Input Form
    # -------------------------------------------------------------
    st.subheader("👤 Customer Profile Configuration")
    
    with st.form(key="churn_prediction_form"):
        # Section 1: Demographics & Account Info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 📋 Account & Demographics")
            customer_id = st.text_input(
                "Customer ID (Optional)",
                value=preset_data.get("customerID", "CUST-7590"),
                help="Optional identifier for reporting; never passed as a model feature.",
            )
            gender = st.selectbox(
                "Gender",
                options=["Female", "Male"],
                index=0 if preset_data.get("gender", "Female") == "Female" else 1,
            )
            senior_citizen_val = preset_data.get("SeniorCitizen", 0)
            senior_citizen = st.selectbox(
                "Senior Citizen",
                options=[0, 1],
                format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
                index=senior_citizen_val,
            )
            partner = st.selectbox(
                "Partner",
                options=["Yes", "No"],
                index=0 if preset_data.get("Partner", "Yes") == "Yes" else 1,
            )
            dependents = st.selectbox(
                "Dependents",
                options=["No", "Yes"],
                index=0 if preset_data.get("Dependents", "No") == "No" else 1,
            )

        with col2:
            st.markdown("##### 📜 Contract & Billing")
            contract_opts = ["Month-to-month", "One year", "Two year"]
            contract = st.selectbox(
                "Contract Term",
                options=contract_opts,
                index=contract_opts.index(preset_data.get("Contract", "Month-to-month")),
            )
            paperless = st.selectbox(
                "Paperless Billing",
                options=["Yes", "No"],
                index=0 if preset_data.get("PaperlessBilling", "Yes") == "Yes" else 1,
            )
            payment_opts = [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
            payment_method = st.selectbox(
                "Payment Method",
                options=payment_opts,
                index=payment_opts.index(preset_data.get("PaymentMethod", "Electronic check")),
            )
            tenure = st.number_input(
                "Tenure (Months with Company)",
                min_value=0.0,
                max_value=120.0,
                value=float(preset_data.get("tenure", 1.0)),
                step=1.0,
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=500.0,
                value=float(preset_data.get("MonthlyCharges", 29.85)),
                step=1.0,
            )
            total_charges = st.number_input(
                "Total Charges ($ Cumulative)",
                min_value=0.0,
                max_value=20000.0,
                value=float(preset_data.get("TotalCharges", 29.85)),
                step=10.0,
            )

        with col3:
            st.markdown("##### 🌐 Subscribed Services")
            phone_service = st.selectbox(
                "Phone Service",
                options=["Yes", "No"],
                index=0 if preset_data.get("PhoneService", "No") == "Yes" else 1,
            )
            multiple_lines_opts = ["No phone service", "No", "Yes"]
            multiple_lines = st.selectbox(
                "Multiple Lines",
                options=multiple_lines_opts,
                index=multiple_lines_opts.index(preset_data.get("MultipleLines", "No phone service")),
            )
            internet_opts = ["DSL", "Fiber optic", "No"]
            internet_service = st.selectbox(
                "Internet Service Provider",
                options=internet_opts,
                index=internet_opts.index(preset_data.get("InternetService", "DSL")),
            )
            security_opts = ["No", "Yes", "No internet service"]
            online_security = st.selectbox(
                "Online Security",
                options=security_opts,
                index=security_opts.index(preset_data.get("OnlineSecurity", "No")),
            )
            backup_opts = ["No", "Yes", "No internet service"]
            online_backup = st.selectbox(
                "Online Backup",
                options=backup_opts,
                index=backup_opts.index(preset_data.get("OnlineBackup", "Yes")),
            )
            device_opts = ["No", "Yes", "No internet service"]
            device_protection = st.selectbox(
                "Device Protection",
                options=device_opts,
                index=device_opts.index(preset_data.get("DeviceProtection", "No")),
            )
            tech_opts = ["No", "Yes", "No internet service"]
            tech_support = st.selectbox(
                "Tech Support",
                options=tech_opts,
                index=tech_opts.index(preset_data.get("TechSupport", "No")),
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                options=["No", "Yes", "No internet service"],
                index=0 if preset_data.get("StreamingTV", "No") == "No" else 1,
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                options=["No", "Yes", "No internet service"],
                index=0 if preset_data.get("StreamingMovies", "No") == "No" else 1,
            )

        submit_btn = st.form_submit_button(
            label="🎯 Predict Churn Risk",
            use_container_width=True,
            type="primary",
        )

    # -------------------------------------------------------------
    # Prediction Execution & Results
    # -------------------------------------------------------------
    if submit_btn:
        payload = {
            "customerID": customer_id if customer_id.strip() else None,
            "gender": gender,
            "SeniorCitizen": int(senior_citizen),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": float(tenure),
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
        }

        with st.spinner("Scoring customer profile via FastAPI model serving..."):
            result, err_msg = request_prediction(payload, API_BASE_URL)

        if err_msg:
            st.error(f"❌ {err_msg}")
        elif result:
            prob = result["churn_probability"]
            pred_class = result["predicted_churn"]
            risk_band = result["risk_band"]
            decision_thresh = result.get("decision_threshold", 0.50)

            st.divider()
            st.subheader("📈 Prediction & Risk Assessment Results")

            # Metrics display
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric(
                    label="Churn Probability",
                    value=f"{prob * 100:.2f}%",
                    delta=f"{'+' if prob >= decision_thresh else '-'}{abs(prob - decision_thresh)*100:.1f}% vs threshold",
                )
            with m_col2:
                status_label = "Likely to Churn" if pred_class == 1 else "Likely to Retain"
                st.metric(label="Predicted Class", value=status_label)
            with m_col3:
                st.metric(label="Assigned Risk Band", value=risk_band)
            with m_col4:
                st.metric(label="Decision Threshold", value=f"{decision_thresh * 100:.0f}%")

            # Risk Tier Alert Box
            if risk_band == "Critical Risk":
                st.error(f"🚨 **{risk_band} Tier:** Customer exhibits imminent churn risk ({prob * 100:.1f}% probability).", icon="🔥")
            elif risk_band == "High Risk":
                st.warning(f"⚠️ **{risk_band} Tier:** Customer exhibits elevated churn risk ({prob * 100:.1f}% probability).", icon="⚡")
            elif risk_band == "Medium Risk":
                st.info(f"ℹ️ **{risk_band} Tier:** Customer exhibits moderate churn risk ({prob * 100:.1f}% probability).", icon="📌")
            else:
                st.success(f"✅ **{risk_band} Tier:** Customer exhibits healthy retention stability ({prob * 100:.1f}% probability).", icon="🛡️")

            # Retention Strategy Card
            st.markdown("#### 💡 Retention Decision Support Guidance")
            st.info(RETENTION_GUIDANCE.get(risk_band, "Review account status."))

    # -------------------------------------------------------------
    # Operational Risk Banding Reference Table
    # -------------------------------------------------------------
    st.divider()
    st.subheader("📚 Operational Churn Risk Tiers Reference")
    st.markdown(
        r"""
        | Risk Band | Probability Range | Strategic Focus | Target Workflow |
        | :--- | :---: | :--- | :--- |
        | 🟢 **Low Risk** | $< 20.0\%$ | General Relationship Care | Standard marketing newsletters; no retention spend required. |
        | 🟡 **Medium Risk** | $20.0\% - < 50.0\%$ | Early Adoption & Engagement | Feature onboarding prompts, survey check-ins, product tips. |
        | 🟠 **High Risk** | $50.0\% - < 75.0\%$ | Proactive Outreach | Account manager check-in, 1-year contract incentive offer. |
        | 🔴 **Critical Risk** | $\ge 75.0\%$ | Emergency Intervention | High-priority outreach, contract renewal credit, executive review. |
        
        *Disclaimer: Operational bands categorize probability ranges for business resource allocation and are not guaranteed deterministic outcomes.*
        """
    )

    # -------------------------------------------------------------
    # Expanders: Model & System Architecture
    # -------------------------------------------------------------
    with st.expander("ℹ️ About the Predictive Model"):
        st.markdown(
            """
            - **Serving Model:** Logistic Regression (Served via FastAPI)
            - **Benchmark Model:** Random Forest (Evaluated in Phase 6)
            - **Transformed Feature Count:** 47 dense numerical features (StandardScaler + OneHotEncoder)
            - **Decision Threshold:** 0.50 (Customizable for business cost matrices)
            - **Data Leakage Safeguards:** Preprocessing pipeline was fitted strictly on training data in Phase 4 and is reused immutably by FastAPI.
            """
        )

    with st.expander("🏗️ System Architecture & Data Flow"):
        st.code(
            """
Customer Input (Streamlit Form)
      │
      ▼ HTTP POST /predict-churn (JSON)
FastAPI Backend (src/api/main.py)
      │
      ├── Load artifacts/preprocessor.joblib (Phase 4 Pipeline)
      │      - Engineer total_services & monthly_charges_diff
      │      - Impute, Standardize & One-Hot Encode ──> (1, 47) array
      │
      ├── Score via artifacts/logistic_regression.joblib (Phase 5 Model)
      │      - Compute P(Churn=1 | X) via predict_proba
      │      - Apply 0.50 Decision Threshold ──> 0 or 1
      │      - Map to Operational Risk Band (Low, Medium, High, Critical)
      │
      ▼ HTTP 200 JSON Response
Streamlit Dashboard (Render Results & Guidance)
            """,
            language="text",
        )


if __name__ == "__main__":
    render_dashboard()
