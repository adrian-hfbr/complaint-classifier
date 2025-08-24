# src/main.py
from fastapi import FastAPI, HTTPException, Depends
from pathlib import Path
import uuid
from contextlib import asynccontextmanager

from .services import ModelService
from .schema import ComplaintRequest, ClassificationResponse

# --- Define the Global Service Variable ---
# will be populated during the startup event.
model_service: ModelService | None = None

# --- Define the Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's startup. Loads the model into the global variable.
    """
    global model_service
    print("▶️ Application startup: Attempting to load ML pipeline...")
    
    try:
        pipeline_path = Path("models/best_pipeline.pkl")
        model_service = ModelService(pipeline_path=pipeline_path)
        print("ML pipeline loaded successfully.")
    except Exception as e:
        model_service = None
        print(f"ERROR: Model loading failed. API will be unavailable. Error: {e}")

    yield
    print("Application shutdown.")

# Create the App and Register the Lifespan Handler
app = FastAPI(title="Complaint Classifier API", lifespan=lifespan)

# --- Dependency Getter ---
def get_model_service() -> ModelService | None:
    """
    This dependency getter now simply returns the global model_service instance.
    """
    return model_service

# --- API Endpoint ---
@app.post("/classify", response_model=ClassificationResponse)
def classify_complaint(
    request: ComplaintRequest,
    service: ModelService = Depends(get_model_service)
):
    """
    Receives a complaint narrative and returns the predicted product category.
    """
    if not service:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded or unavailable."
        )
        
    try:
        prediction = service.predict(request.narrative)
        
        return ClassificationResponse(
            predicted_product=prediction,
            request_id=uuid.uuid4()
        )
    except Exception as e:
        print(f"ERROR: Prediction failed. {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")