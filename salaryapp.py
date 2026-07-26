from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os
import traceback

# =====================================
# Create FastAPI App
# =====================================

app = FastAPI(title="Employee Salary Predictor API")

# =====================================
# Enable CORS
# =====================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL after deployment for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# Load Trained Model
# =====================================

model = None

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "salary_prediction_model.pkl"
)

try:
    print(f"Looking for model at: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")

except Exception:
    print("❌ Error loading model:")
    traceback.print_exc()
    model = None

# =====================================
# Input Schema
# =====================================

class EmployeeData(BaseModel):
    age: int
    gender: str
    education_level: str
    years_of_experience: float
    job_title: str

# =====================================
# Home Route
# =====================================

@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Employee Salary Predictor API is Running!"
    }

# =====================================
# Health Check Route
# =====================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

# =====================================
# Prediction Route
# =====================================

@app.post("/predict")
def predict(data: EmployeeData):

    if model is None:
        return {
            "status": "error",
            "message": "Model could not be loaded. Check Render logs."
        }

    try:
        # Create DataFrame with the same column names used during training
        input_df = pd.DataFrame({
            "Age": [data.age],
            "Gender": [data.gender],
            "Education Level": [data.education_level],
            "Years of Experience": [data.years_of_experience],
            "Job Title": [data.job_title]
        })

        print("\nReceived Input:")
        print(input_df)

        prediction = model.predict(input_df)

        return {
            "status": "success",
            "predicted_salary": round(float(prediction[0]), 2)
        }

    except Exception as e:
        print("Prediction Error:")
        traceback.print_exc()

        return {
            "status": "error",
            "message": str(e)
        }

# =====================================
# Note:
# Do NOT include uvicorn.run() when deploying to Render.
#
# Render Start Command:
# uvicorn backend:app --host 0.0.0.0 --port $PORT
#
# Replace 'backend' with your Python filename if it is different.
# =====================================