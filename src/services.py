# src/services.py
import joblib
from pathlib import Path
from .preprocessing import ComplaintPreprocessor

class ModelService:
    """
    Encapsulates the ML pipeline loading and prediction logic.
    """
    def __init__(self, pipeline_path: Path):
        """
        Initializes the service by loading the pipeline from the specified path.

        This follows the "fail fast" principle: if the pipeline artifact is missing
        or corrupt, the service will fail to start, which is better than failing
        later during a user request.

        Args:
            pipeline_path (Path): The path to the saved best_pipeline.pkl artifact.
        """
        self.pipeline_path = pipeline_path
        try:
            # Load pipeline
            self.pipeline = joblib.load(self.pipeline_path)
        except FileNotFoundError:
            print(f"ERROR: Pipeline artifact not found at {self.pipeline_path}")
            raise
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while loading the pipeline: {e}")
            raise

    def predict(self, narrative: str) -> str:
        """
        Makes a prediction on a single, raw complaint narrative.

        Args:
            narrative (str): The raw text of the customer complaint.

        Returns:
            str: The predicted product category as a string.
        """
        # The scikit-learn pipeline's .predict() method expects an iterable
        # (like a list or Series), not a single string. We wrap the narrative
        # in a list to create the correct input format.
        prediction_array = self.pipeline.predict([narrative])
        
        # The prediction comes back as an array with a single element.
        # We extract that single prediction to return a clean string.
        return prediction_array[0]