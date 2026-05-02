import streamlit as st
import requests
import pandas as pd

# 1. Page Config
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

st.title("🩺 Diabetes Predictor")
st.write("Enter the patient's health metrics below to check the risk level.")

# 2. Input Fields (using columns for a cleaner look)
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.slider("Glucose Level", 0, 200, 120)
    bp = st.slider("Blood Pressure", 0, 140, 70)
    skin = st.slider("Skin Thickness", 0, 100, 20)

with col2:
    insulin = st.slider("Insulin Level", 0, 900, 80)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0)
    pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.number_input("Age", min_value=21, max_value=100, value=30)

# 3. Prediction Logic
if st.button("Predict Risk"):
    # Create the JSON payload (MUST match your FastAPI Pydantic model)
    payload = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age
    }

    # Replace with your ACTUAL Render URL
    API_URL = "https://diabetes-detector-ali2gitz.onrender.com/predict"
    
    try:
        response = requests.post(API_URL, json=payload)
        
        # 1. Check if the API request was successful (Status Code 200)
        if response.status_code == 200:
            prediction_data = response.json()
            
            # 2. Safely get the diagnosis
            diagnosis = prediction_data.get('diagnosis', 'Unknown Result')
            
            st.subheader("Result:")
            if diagnosis == "Diabetic":
                st.error(f"Prediction: {diagnosis}")
            else:
                st.success(f"Prediction: {diagnosis}")
                #NEW Code
            st.divider()
            st.subheader("📊 What influenced this prediction?")
            
            # Get the importance data from the API response
            importances = prediction_data.get('feature_importance', {})
            
            if importances:
            # Convert to a DataFrame for easy plotting
                df_importance = pd.DataFrame({
                    'Feature': importances.keys(),
                    'Importance': importances.values()
                }).sort_values(by='Importance', ascending=True)

                # Create the horizontal bar chart
                st.bar_chart(data=df_importance, x='Feature', y='Importance', horizontal=True)
                
                st.info("The chart above shows which health markers the model weighed most heavily for this dataset.")
                #End of new code
        else:
            # 3. If API failed, show the error detail
            error_detail = response.json().get('detail', 'Unknown API Error')
            st.warning(f"API Error ({response.status_code}): {error_detail}")
            
    except Exception as e:
        st.error(f"Could not reach the server: {e}")