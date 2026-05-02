from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
# 2. Load the saved model
model = joblib.load("diabetes_model.joblib")
#for importance features
# At the top of main.py, get the importance scores once
feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Pedigree", "Age"]
importances = model.feature_importances_.tolist()
feature_importance_dict = dict(zip(feature_names, importances))

app = FastAPI()
app = FastAPI()

# 3Define the input shape (The 8 features of the Pima dataset)
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int
@app.post("/predict")
def predict_diabetes(data: DiabetesInput):
    input_data = [[data.Pregnancies, data.Glucose, data.BloodPressure, 
                    data.SkinThickness, data.Insulin, data.BMI, 
                    data.DiabetesPedigreeFunction, data.Age]]
    
    prediction = model.predict(input_data)
    result = "Diabetic" if int(prediction[0]) == 1 else "Healthy"
    
    return {
        "prediction": int(prediction[0]),
        "diagnosis": result,
        "feature_importance": feature_importance_dict # key matches app.py
    }
#end code for important features
# 1. Initialize FastAPI app

@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running!"}

# 4. Create the POST /predict endpoint
@app.post("/predict")
def predict_diabetes(data: DiabetesInput):
    # Convert input data to a list of values
    input_data = [[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]]
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Return the result
    result = "Diabetic" if int(prediction[0]) == 1 else "Healthy"
    return {
        "prediction": int(prediction[0]),
        "diagnosis": result
    }