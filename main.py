from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# 1. Initialize FastAPI app
app = FastAPI()

# 2. Load the saved model

model = joblib.load("diabetes_model.joblib")

# 3. Define the input shape (The 8 features of the Pima dataset)
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

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