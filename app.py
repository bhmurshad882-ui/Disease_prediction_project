import streamlit as st
import requests
import pandas as pd

# Page config
st.set_page_config(
    page_title="AI Disease Predictor",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }
    
    .main-header {
        text-align: center;
        padding: 30px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        backdrop-filter: blur(10px);
        margin-bottom: 40px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .prediction-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        animation: fadeIn 1s ease-in-out;
    }

    .disease-name {
        font-size: 3rem;
        font-weight: 800;
        color: #ff4b4b;
        margin: 20px 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .confidence-score {
        font-size: 1.5rem;
        color: #00ff87;
    }

    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* Input Styling overrides */
    .stNumberInput > div > div > input {
        color: #fff !important;
    }
</style>
""", unsafe_allow_html=True)

# Main layout
st.markdown('<div class="main-header"><h1>🧬 AI Disease Predictor</h1><p>Enter patient blood test parameters to predict potential diseases using an advanced Artificial Neural Network.</p></div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Patient Vitals & Blood Chemistry")
    st.write("Please fill in the 24 key parameters below:")
    
    # Feature inputs organized in columns
    features = [
        "Glucose", "Cholesterol", "Hemoglobin", "Platelets", "White Blood Cells", 
        "Red Blood Cells", "Hematocrit", "Mean Corpuscular Volume", 
        "Mean Corpuscular Hemoglobin", "Mean Corpuscular Hemoglobin Concentration", 
        "Insulin", "BMI", "Systolic Blood Pressure", "Diastolic Blood Pressure", 
        "Triglycerides", "HbA1c", "LDL Cholesterol", "HDL Cholesterol", 
        "ALT", "AST", "Heart Rate", "Creatinine", "Troponin", "C-reactive Protein"
    ]
    
    input_data = {}
    
    c1, c2, c3 = st.columns(3)
    
    # We distribute inputs nicely across 3 columns
    for idx, feature in enumerate(features):
        col_to_use = [c1, c2, c3][idx % 3]
        with col_to_use:
            input_data[feature] = st.number_input(feature, value=0.5, step=0.01, format="%.4f")

with col2:
    st.markdown("<br><br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Run AI Prediction", use_container_width=True, type="primary")
    
    if predict_btn:
        with st.spinner("Analyzing patient data with Neural Network..."):
            try:
                # Send to backend
                res = requests.post("http://127.0.0.1:5000/predict", json=input_data)
                
                if res.status_code == 200:
                    result = res.json()
                    disease = result.get('prediction', 'Unknown')
                    confidence = result.get('confidence', 0)
                    
                    st.markdown(f"""
                        <div class="prediction-card">
                            <h3>Diagnosis Result</h3>
                            <div class="disease-name">{disease}</div>
                            <div class="confidence-score">Confidence: {confidence}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if disease != 'Healthy':
                        st.warning("⚠️ Warning: Patient requires immediate medical review based on the AI prediction.")
                    else:
                        st.success("✅ Patient appears healthy based on current metrics.")
                else:
                    st.error(f"Error communicating with backend API. Status Code: {res.status_code}")
                    st.json(res.json())
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to the backend server. Please make sure the Flask backend is running on port 5000.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
