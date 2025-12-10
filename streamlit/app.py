import streamlit as st
import requests
import os

# Get API URL from environment variable
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Simple Streamlit Demo", page_icon="🚀")

st.title("🚀 Simple Streamlit ↔️ FastAPI Demo")
st.markdown("---")

st.write("This is a simple demo showing how Streamlit communicates with FastAPI.")
st.write(f"**API URL:** `{API_URL}`")

st.markdown("---")

# Input box
user_input = st.text_input(
    "Enter some text:",
    placeholder="Type something here...",
    help="Enter any text and click Submit to send it to the FastAPI backend"
)

# Submit button
if st.button("Submit", type="primary"):
    if user_input:
        with st.spinner("Sending request to FastAPI..."):
            try:
                # Send POST request to FastAPI
                response = requests.post(
                    f"{API_URL}/process",
                    json={"text": user_input},
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success("✅ Response received from FastAPI!")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Original Text", data["original"])
                        st.metric("Text Length", data["length"])
                    
                    with col2:
                        st.metric("Reversed Text", data["reversed"])
                        st.metric("Uppercase Text", data["uppercase"])
                    
                    # Show raw JSON response
                    with st.expander("View Raw JSON Response"):
                        st.json(data)
                else:
                    st.error(f"❌ Error: Received status code {response.status_code}")
                    st.write(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to FastAPI. Is it running?")
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    else:
        st.warning("⚠️ Please enter some text first!")

st.markdown("---")

# Health check section
st.subheader("🏥 API Health Check")
if st.button("Check API Health"):
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success(f"✅ API is healthy! Response: {response.json()}")
        else:
            st.error(f"❌ API returned status code: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Cannot reach API: {str(e)}")

# Footer
st.markdown("---")
st.caption("This demo shows the basic connection between Streamlit and FastAPI. The models folder and data folder are mounted and ready for future use.")