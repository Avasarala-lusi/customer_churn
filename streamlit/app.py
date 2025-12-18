import json
import os
from pathlib import Path
from typing import Any, Dict

import requests
import streamlit as st

# -----------------------------------------------------------------------------
# MUST be the first Streamlit command
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="🏠", layout="centered")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SCHEMA_PATH = Path("/app/data/data_schema.json")

# API_URL is set in docker-compose environment
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"

# -----------------------------------------------------------------------------
# Load schema from JSON file
# -----------------------------------------------------------------------------
@st.cache_resource
def load_schema(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


schema = load_schema(SCHEMA_PATH)

numerical_features = schema.get("numerical", {})
categorical_features = schema.get("categorical", {})

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("🏦 Customer Churn Prediction")
st.write(
    f"This app predicts whether a customer will churn based on their profile. "
    f"The prediction is powered by FastAPI backend at **{API_BASE_URL}**."
)

st.header("Customer Information")

user_input: Dict[str, Any] = {}

# -----------------------------------------------------------------------------
# Numerical Features
# -----------------------------------------------------------------------------
st.subheader("📊 Numerical Features") 

# Decide which features use sliders
SLIDER_FEATURES = {"creditScore", "age", "tenure", "numofProducts"}

for feature_name, stats in numerical_features.items():
    min_val = float(stats.get("min", 0.0))
    max_val = float(stats.get("max", 1000.0))
    mean_val = float(stats.get("mean", (min_val + max_val) / 2))
    median_val = float(stats.get("median", mean_val))

    # Use median as default
    default_val = median_val

    label = feature_name.replace("_", " ").title()
    help_text = (
        f"Min: {min_val:.2f}, Max: {max_val:.2f}, "
        f"Mean: {mean_val:.2f}, Median: {median_val:.2f}"
    )

    if feature_name in SLIDER_FEATURES:
        # Determine step size based on range and semantics
        if feature_name in {"creditScore"}:
            step = 10.0  # Credit score increments
        elif feature_name in {"age", "tenure", "numofProducts"}:
            step = 1.0 # Integer values
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
        # Use number_input for large range features like balance and salary
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
# -----------------------------------------------------------------------------
# Categorical Features
# -----------------------------------------------------------------------------
st.subheader("👤 Categorical Features")

for feature_name, info in categorical_features.items():
    unique_values = info.get("unique_values", [])
    value_counts = info.get("value_counts", {})

    if not unique_values:
        continue

    # Default to the most common value
    if value_counts:
        default_value = max(value_counts, key=value_counts.get)
    else:
        default_value = unique_values[0]

    try:
        default_idx = unique_values.index(default_value)
    except ValueError:
        default_idx = 0

    label = feature_name.replace("_", " ").title()

    # Better UI for binary features (0/1)
    if set(unique_values) == {0, 1} or set(unique_values) == {1, 0}:
        # Use radio buttons for binary choices
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

        # Find the default index based on the default_value
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
        # Map selection back to numeric value
        selected_idx = display_options.index(selected)
        user_input[feature_name] = option_values[selected_idx]
    else:
        # Use selectbox for multi-value categorical features
        user_input[feature_name] = st.selectbox(
            label,
            options=unique_values,
            index=default_idx,
            key=feature_name,
            help=f"Distribution: {value_counts}",
        )

st.markdown("---")

# -----------------------------------------------------------------------------
# Predict Button
# -----------------------------------------------------------------------------
if st.button("🔮 Predict", type="primary"):
    payload = {"instances": [user_input]}

    with st.spinner("Calling API for prediction..."):
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
                    st.success("✅ Prediction completed!")

                    st.subheader("Prediction Result")

                    # Display churn prediction with nice formatting
                    if pred == 1:
                        st.error("⚠️ **HIGH RISK**: Customer is likely to churn")
                        st.markdown(
                            "**Recommendation:** Consider retention strategies such as "
                            "personalized offers, loyalty programs, or proactive customer support."
                        )
                    else:
                        st.success("✅ **LOW RISK**: Customer is likely to stay")
                        st.markdown(
                            "**Recommendation:** Continue providing excellent service to maintain "
                            "customer satisfaction and loyalty."
                        )
                     # Show input summary in expander
                    with st.expander("📋 View Customer Profile"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Numerical Features**")
                            for k, v in user_input.items():
                                if k in numerical_features:
                                    st.write(f"- **{k}**: {v:,.2f}")
                        
                        with col2:
                            st.markdown("**Categorical Features**")
                            for k, v in user_input.items():
                                if k in categorical_features:
                                    st.write(f"- **{k}**: {v}")

st.markdown("---")
st.caption(
    f"📁 Schema: `{SCHEMA_PATH}`  \n"
    f"🌐 API: `{API_BASE_URL}`"
)