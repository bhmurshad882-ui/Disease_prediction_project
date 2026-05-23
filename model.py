import joblib
import os
import numpy as np
import pandas as pd

# Load artifacts
MODEL_DIR = os.path.dirname(__file__)
model_path = os.path.join(MODEL_DIR, 'disease_ann_model.pkl')
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
encoder_path = os.path.join(MODEL_DIR, 'encoder.pkl')

# Load globally to avoid reloading on every request
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)

def predict_disease(features: dict):
    """
    Predict disease from dictionary of features.
    The features dict must have keys matching the original dataset (except 'Disease').
    """
    # Create DataFrame to ensure column order matches training data
    df = pd.DataFrame([features])
    
    # Scale features
    df_scaled = scaler.transform(df)
    
    # Predict probabilities
    probs = model.predict_proba(df_scaled)[0]
    
    # Get highest probability
    pred_idx = np.argmax(probs)
    confidence = float(probs[pred_idx])
    
    # Get disease name
    disease_name = encoder.inverse_transform([pred_idx])[0]
    
    return {
        "prediction": str(disease_name),
        "confidence": round(confidence * 100, 2)
    }
