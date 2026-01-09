import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Ad Purchase Predictor", page_icon="🛍️")
st.title("🛍️ Social Network Ad Purchase Predictor")

# Load Assets
if os.path.exists('social_ads_model.pkl'):
    model = joblib.load('social_ads_model.pkl')
    scaler = joblib.load('social_ads_scaler.pkl')
    gender_le = joblib.load('gender_encoder.pkl')

    st.write("Predict if a customer will buy the product based on their profile.")

    # User Inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 70, 30)
    salary = st.number_input("Estimated Annual Salary ($)", value=50000)

    if st.button("Predict Purchase"):
        # Process inputs
        gender_num = gender_le.transform([gender])[0]
        features = np.array([[gender_num, age, salary]])
        
        # Scale and Predict
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features)[0][1] * 100

        if prediction[0] == 1:
            st.success(f"✅ Likely to Purchase! (Confidence: {probability:.1f}%)")
        else:
            st.error(f"❌ Unlikely to Purchase. (Confidence: {100-probability:.1f}%)")
else:
    st.error("Missing model files. Please upload the .pkl files to GitHub.")