# src/main.py
from fastapi import FastAPI, HTTPException, Depends
from pathlib import Path
import uuid
from contextlib import asynccontextmanager

from .services import ModelService
from .schema import ComplaintRequest, ClassificationResponse, CorrectionRequest

from prometheus_client import Counter, Histogram
from starlette.responses import Response
import prometheus_client

# --- 1. Metrics For Monitoring ---
# To understand the systems behavior
PREDICTIONS = Counter(
    "predictions_total", "Total number of predictions made.", ["predicted_product"]
)
ERRORS = Counter(
    "prediction_errors_total", "Total number of errors encountered during prediction."
)
LATENCY = Histogram(
    "prediction_latency_seconds", "Histogram of prediction latency in seconds."
)


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


# --- /metrics Endpoint ---
@app.get("/metrics")
def get_metrics():
    """
    Endpoint for Prometheus to scrape metrics.
    """
    return Response(
        media_type="text/plain", content=prometheus_client.generate_latest()
    )


# --- API Endpoint ---
@app.post("/classify", response_model=ClassificationResponse)
def classify_complaint(
    request: ComplaintRequest, service: ModelService = Depends(get_model_service)
):
    """
    Receives a complaint narrative and returns the predicted product category.
    """
    with LATENCY.time():
        try:
            prediction = service.predict(request.narrative)

            # Increment the prediction counter, using the prediction as a label
            PREDICTIONS.labels(predicted_product=prediction).inc()

            return ClassificationResponse(
                predicted_product=prediction, request_id=uuid.uuid4()
            )
        except Exception as e:
            # Increment the error counter if something goes wrong
            ERRORS.inc()
            print(f"ERROR: Prediction failed. {e}")
            raise HTTPException(status_code=500, detail="Model inference failed.")


@app.post("/correct")
def correct_prediction(request: CorrectionRequest):
    """
    Endpoint to receive a corrected label for a previous prediction.
    In a real application, this would save the data to a database.
    """
    # Print to demonstrate mechanism
    print("\n--- Correction Received ---")
    print(f"  Request ID: {request.request_id}")
    print(f"   Narrative: {request.narrative[:50]}...")
    print(f"   Predicted: {request.predicted_product}")
    print(f"     Correct: {request.correct_product}")
    print("---------------------------\n")

    # In a PROD-system, this would be saved in a DB
    # or a message queue for later use in retraining.
    return {"status": "Correction received"}
