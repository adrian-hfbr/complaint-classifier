# src/main.py
from fastapi import FastAPI, HTTPException
from pathlib import Path
import uuid

# imports encapsulated ML logic
from .services import ModelService

# imports API contract
from .schema import ComplaintRequest, ClassificationResponse

# Create the FastAPI app instance -> main object that runs on web server
app = FastAPI(title="Complaint Classifier API")

PIPELINE_PATH = Path("models/best_pipeline.pkl")

# --- Load the Model at Startup ---
# Create a single, global instance of the ModelService
# Loaded once into memory and not for every request
model_service = ModelService(pipeline_path=PIPELINE_PATH)


# --- Define the API Endpoint ---
# Decorator that turns function into endpoint
@app.post("/classify", response_model=ClassificationResponse)
def classify_complaint(request: ComplaintRequest):
    """
    Receives a complaint narrative, checks against configured schema
    and returns the predicted product category.
    """
    try:
        prediction = model_service.predict(request.narrative)

        # Takes prediction result and creates CR object
        return ClassificationResponse(
            predicted_product=prediction, request_id=uuid.uuid4()
        )
    except Exception as e:
        print(f"Prediction failed. {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")
