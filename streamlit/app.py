import json
import os
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

import requests
import streamlit as st

# -----------------------------------------------------------------------------
# MUST be the first Streamlit command
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction", 
    page_icon="🦆", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom CSS for better styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .risk-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .risk-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    .api-status-online {
        color: #00c853;
        font-weight: bold;
    }
    .api-status-offline {
        color: #ff5252;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SCHEMA_PATH = Path("/app/data/data_schema.json")
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"

# -----------------------------------------------------------------------------
# Initialize session state
# -----------------------------------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = 0
if "active_preset" not in st.session_state:
    st.session_state.active_preset = None

# -----------------------------------------------------------------------------
# Load schema from JSON file
# -----------------------------------------------------------------------------
@st.cache_resource
def load_schema(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

# -----------------------------------------------------------------------------
# API Health Check
# -----------------------------------------------------------------------------
def check_api_health() -> bool:
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return resp.status_code == 200
    except:
        try:
            resp = requests.get(API_BASE_URL, timeout=3)
            return resp.status_code in [200, 404]
        except:
            return False

schema = load_schema(SCHEMA_PATH)
numerical_features = schema.get("numerical", {})
categorical_features = schema.get("categorical", {})

# -----------------------------------------------------------------------------
# Preset Profiles
# -----------------------------------------------------------------------------
PRESET_PROFILES = {
    "Low Risk": {
        "creditScore": 750.0,
        "age": 45.0,
        "tenure": 8.0,
        "balance": 100000.0,
        "numofProducts": 2.0,
        "hasCrCard": 1,
        "isActiveMember": 1,
        "estimatedSalary": 120000.0,
        "isZeroBalance": 0,
        "gender": "Female",
        "geography": "France"
    },
    "High Risk": {
        "creditScore": 740.0,
        "age": 54.0,
        "tenure": 8.0,
        "balance": 126418,
        "numofProducts": 1.0,
        "hasCrCard": 1,
        "isActiveMember": 0,
        "estimatedSalary": 134420.0,
        "isZeroBalance": 1,
        "gender":"Male",
        "geography": "Germany"
    }
}

def apply_preset(profile_type: str):
    """Apply preset values to session state"""
    # Clear existing session state first
    for key in list(st.session_state.keys()):
        if key in numerical_features or key in categorical_features:
            del st.session_state[key]
    
    # Set the active preset
    st.session_state.active_preset = profile_type
    st.session_state.reset_trigger += 1

# -----------------------------------------------------------------------------
# Main App Layout
# -----------------------------------------------------------------------------
col_header1, col_header2 = st.columns([3, 1])

with col_header1:
    st.markdown('<p class="main-header">🦆 Customer Churn Prediction</p>', unsafe_allow_html=True)
    st.write("Predict customer churn risk using machine learning powered by FastAPI backend")

with col_header2:
    api_online = check_api_health()
    if api_online:
        st.markdown('🟢 <span class="api-status-online">API Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('🔴 <span class="api-status-offline">API Offline</span>', unsafe_allow_html=True)
    st.caption(f"Endpoint: `{API_BASE_URL}`")

st.markdown("---")

# -----------------------------------------------------------------------------
# Sidebar for Inputs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📋 Customer Profile Input")
    
    # Quick preset buttons
    st.subheader("⚡ Quick Presets")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🟢 Low Risk Profile", use_container_width=True):
            apply_preset("Low Risk")
            st.rerun()
    with col2:
        if st.button("🔴 High Risk Profile", use_container_width=True):
            apply_preset("High Risk")
            st.rerun()
    
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key in numerical_features or key in categorical_features:
                del st.session_state[key]
        st.session_state.active_preset = None
        st.session_state.reset_trigger += 1
        st.rerun()
    
    st.markdown("---")
    
    user_input: Dict[str, Any] = {}
    
    # Numerical Features
    st.subheader("📊 Numerical Features")
    
    SLIDER_FEATURES = {"creditScore", "age", "tenure", "numofProducts"}
    
    for feature_name, stats in numerical_features.items():
        min_val = float(stats.get("min", 0.0))
        max_val = float(stats.get("max", 1000.0))
        mean_val = float(stats.get("mean", (min_val + max_val) / 2))
        median_val = float(stats.get("median", mean_val))
        
        # Use preset value if active, otherwise use median
        if st.session_state.active_preset and feature_name in PRESET_PROFILES[st.session_state.active_preset]:
            default_val = float(PRESET_PROFILES[st.session_state.active_preset][feature_name])
        else:
            default_val = median_val
        
        label = feature_name.replace("_", " ").title()
        help_text = (
            f"Min: {min_val:.2f}, Max: {max_val:.2f}, "
            f"Mean: {mean_val:.2f}, Median: {median_val:.2f}"
        )
        
        if feature_name in SLIDER_FEATURES:
            if feature_name in {"creditScore"}:
                step = 10.0
            elif feature_name in {"age", "tenure", "numofProducts"}:
                step = 1.0
            else:
                step = 0.1
            
            user_input[feature_name] = st.slider(
                label,
                min_value=min_val,
                max_value=max_val,
                value=float(default_val),
                step=step,
                help=help_text,
                key=feature_name,
            )
        else:
            range_val = max_val - min_val
            if range_val > 100000:
                step = 1000.0
            elif range_val > 10000:
                step = 100.0
            elif range_val > 1000:
                step = 10.0
            elif range_val > 100:
                step = 1.0
            else:
                step = 0.01
            
            user_input[feature_name] = st.number_input(
                label,
                min_value=min_val,
                max_value=max_val,
                value=float(default_val),
                step=step,
                help=help_text,
                key=feature_name,
            )
    
    st.markdown("---")
    
    # Categorical Features
    st.subheader("👤 Categorical Features")
    
    for feature_name, info in categorical_features.items():
        unique_values = info.get("unique_values", [])
        value_counts = info.get("value_counts", {})
        
        if not unique_values:
            continue
        
        # Use preset value if active, otherwise use most common value
        if st.session_state.active_preset and feature_name in PRESET_PROFILES[st.session_state.active_preset]:
            default_value = PRESET_PROFILES[st.session_state.active_preset][feature_name]
        elif value_counts:
            default_value = max(value_counts, key=value_counts.get)
        else:
            default_value = unique_values[0]
        
        try:
            default_idx = unique_values.index(default_value)
        except ValueError:
            default_idx = 0
        
        label = feature_name.replace("_", " ").title()
        
        if set(unique_values) == {0, 1} or set(unique_values) == {1, 0}:
            if feature_name == "hasCrCard":
                display_options = ["No Credit Card", "Has Credit Card"]
                option_values = [0, 1]
            elif feature_name == "isActiveMember":
                display_options = ["Inactive", "Active Member"]
                option_values = [0, 1]
            elif feature_name == "isZeroBalance":
                display_options = ["Has Balance", "Zero Balance"]
                option_values = [0, 1]
            else:
                display_options = ["No", "Yes"]
                option_values = [0, 1]
            
            try:
                default_radio_idx = option_values.index(default_value)
            except (ValueError, TypeError):
                default_radio_idx = 0
            
            selected = st.radio(
                label,
                options=display_options,
                index=default_radio_idx,
                horizontal=True,
                key=feature_name,
                help=f"Distribution: {value_counts}",
            )
            selected_idx = display_options.index(selected)
            user_input[feature_name] = option_values[selected_idx]
        else:
            user_input[feature_name] = st.selectbox(
                label,
                options=unique_values,
                index=default_idx,
                key=feature_name,
                help=f"Distribution: {value_counts}",
            )
    
    st.markdown("---")
    
    # Predict Button in Sidebar
    predict_button = st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Main Content Area - Prediction Results
# -----------------------------------------------------------------------------

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 History", "ℹ️ Info"])

with tab1:
    if predict_button:
        if not api_online:
            st.error("❌ API is offline. Please check the connection.")
        else:
            payload = {"instances": [user_input]}
            
            with st.spinner("🔄 Calling API for prediction..."):
                try:
                    resp = requests.post(PREDICT_ENDPOINT, json=payload, timeout=30)
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Request to API failed: {e}")
                else:
                    if resp.status_code != 200:
                        st.error(f"❌ API error: HTTP {resp.status_code} - {resp.text}")
                    else:
                        data = resp.json()
                        preds = data.get("predictions", [])
                        
                        if not preds:
                            st.warning("⚠️ No predictions returned from API.")
                        else:
                            pred = preds[0]
                            
                            # Save to history
                            st.session_state.prediction_history.append({
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "prediction": pred,
                                "inputs": user_input.copy()
                            })
                            
                            # Display prediction with enhanced styling
                            if pred == 1:
                                st.markdown("""
                                <div class="metric-card risk-high">
                                    <h2>⚠️ HIGH RISK</h2>
                                    <p style="font-size: 1.2rem;">Customer is likely to churn</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.error("### 🎯 Recommended Actions")
                                st.markdown("""
                                - **Immediate Outreach**: Contact customer within 24-48 hours
                                - **Personalized Offers**: Provide tailored retention incentives
                                - **Loyalty Programs**: Enroll in premium rewards program
                                - **Customer Support**: Assign dedicated account manager
                                - **Feedback Survey**: Understand pain points and concerns
                                """)
                            else:
                                st.markdown("""
                                <div class="metric-card risk-low">
                                    <h2>✅ LOW RISK</h2>
                                    <p style="font-size: 1.2rem;">Customer is likely to stay</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.success("### 🎯 Recommended Actions")
                                st.markdown("""
                                - **Maintain Service**: Continue excellent customer experience
                                - **Upsell Opportunities**: Introduce premium products/services
                                - **Engagement Programs**: Keep customer engaged with updates
                                - **Referral Incentives**: Encourage customer referrals
                                - **Satisfaction Checks**: Periodic satisfaction surveys
                                """)
                            
                            # Display customer profile summary
                            st.markdown("---")
                            st.subheader("📋 Customer Profile Summary")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Credit Score", f"{user_input.get('creditScore', 0):.0f}")
                                st.metric("Age", f"{user_input.get('age', 0):.0f}")
                            
                            with col2:
                                st.metric("Tenure (years)", f"{user_input.get('tenure', 0):.0f}")
                                st.metric("Products", f"{user_input.get('numofProducts', 0):.0f}")
                            
                            with col3:
                                st.metric("Balance", f"${user_input.get('balance', 0):,.2f}")
                                active = "Yes" if user_input.get('isActiveMember', 0) == 1 else "No"
                                st.metric("Active Member", active)
                            
                            with col4:
                                st.metric("Salary", f"${user_input.get('estimatedSalary', 0):,.2f}")
                                card = "Yes" if user_input.get('hasCrCard', 0) == 1 else "No"
                                st.metric("Has Credit Card", card)
    else:
        # Initial state - show instructions
        st.info("👈 **Enter customer information in the sidebar and click 'Predict Churn Risk' to get started**")
        
        st.markdown("### 🎯 How It Works")
        st.markdown("""
        1. **Input Customer Data**: Fill in the customer's profile information in the sidebar
        2. **Quick Presets**: Use preset buttons to test with typical low/high risk profiles
        3. **Get Prediction**: Click the predict button to analyze churn risk
        4. **Review Results**: View risk assessment and recommended actions
        5. **Check History**: See all previous predictions in the History tab
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Features Used")
            st.markdown("""
            - Credit Score
            - Age & Tenure
            - Account Balance
            - Product Holdings
            - Activity Status
            - And more...
            """)
        
        with col2:
            st.markdown("### 🎓 Model Benefits")
            st.markdown("""
            - Real-time predictions
            - Actionable insights
            - Risk-based recommendations
            - Easy-to-use interface
            - Scalable architecture
            """)

with tab2:
    st.subheader("📈 Prediction History")
    
    if st.session_state.prediction_history:
        st.write(f"**Total Predictions:** {len(st.session_state.prediction_history)}")
        
        # Display history in reverse chronological order
        for idx, record in enumerate(reversed(st.session_state.prediction_history)):
            with st.expander(f"🕐 {record['timestamp']} - {'🔴 High Risk' if record['prediction'] == 1 else '🟢 Low Risk'}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Numerical Features**")
                    for k, v in record['inputs'].items():
                        if k in numerical_features:
                            st.write(f"- **{k}**: {v:,.2f}")
                
                with col2:
                    st.markdown("**Categorical Features**")
                    for k, v in record['inputs'].items():
                        if k in categorical_features:
                            st.write(f"- **{k}**: {v}")
        
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("No predictions yet. Make your first prediction to see it here!")

with tab3:
    st.subheader("ℹ️ Application Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 Configuration")
        st.code(f"""
Schema Path: {SCHEMA_PATH}
API Base URL: {API_BASE_URL}
Predict Endpoint: {PREDICT_ENDPOINT}
        """)
        
        st.markdown("### 📊 Feature Statistics")
        st.write(f"- Numerical Features: **{len(numerical_features)}**")
        st.write(f"- Categorical Features: **{len(categorical_features)}**")
    
    with col2:
        st.markdown("### 🎯 About")
        st.markdown("""
        This application uses machine learning to predict customer churn risk based on various customer attributes and behaviors.
        
        **Key Capabilities:**
        - Real-time predictions via API
        - Interactive data input
        - Preset customer profiles
        - Prediction history tracking
        - Actionable recommendations
        """)

st.markdown("---")
st.caption("🦆 Customer Churn Prediction System | Powered by Streamlit & FastAPI")
