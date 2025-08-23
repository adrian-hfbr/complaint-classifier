# src/train.py
import argparse # Allows the script to accept command-line arguments
import pandas as pd
import joblib # Used to save trained ML model & preprocessor
import mlflow # Used to log things like model parameters, performance metrics, and the model itself
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Importing my custom preprocessor
from preprocessing import ComplaintPreprocessor

from utils import get_git_commit_hash


def run_training(data_path: str, model_output_path: str):
    """
    Orchestrates the model training and logging pipeline.
    """
    # Start an MLflow run to log all aspects of this training session
    with mlflow.start_run():
        # --- 1. Load and Preprocess Data ---
        df = pd.read_csv(
            data_path,
            nrows=30000, # Use a subsample for rapid development
            low_memory=False # Silences the DtypeWarning
        ) # Load raw data
        preprocessor = ComplaintPreprocessor() # Initialize preprocess-object
        preprocessor.fit(df) # fit preprocess-object on data
        X = preprocessor.transform(df) # create transformed matrix X & y
        y = preprocessor.get_target(df)

        # --- 2. Split Data ---
        # Use a fixed random_state for reproducibility
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # --- 3. Train Model ---
        # Define model with a fixed random_state for deterministic results
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # --- 4. Evaluate Model ---
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1_macro = f1_score(y_test, predictions, average='macro')

        # --- 5. Log to MLflow ---
        # Log the current git commit hash for full reproducibility
        mlflow.log_param("git_commit_hash", get_git_commit_hash())

        # Log model parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)

        # --- 6. Save Artifacts ---
        # Save the fitted preprocessor and model
        joblib.dump(preprocessor, f"{model_output_path}/preprocessor.pkl")
        joblib.dump(model, f"{model_output_path}/model.pkl")

        # Log the artifacts to MLflow for a complete record
        mlflow.log_artifact(f"{model_output_path}/preprocessor.pkl")
        mlflow.log_artifact(f"{model_output_path}/model.pkl")

        print(f"Training complete. Accuracy: {accuracy:.4f}, F1 (Macro): {f1_macro:.4f}")

if __name__ == "__main__":
    # --- 7. Command-Line Interface ---
    parser = argparse.ArgumentParser(description="Train a complaint classification model.")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the gzipped CSV data file."
    )
    parser.add_argument(
        "--model-output-path",
        type=str,
        required=True,
        help="Directory to save the trained model and preprocessor artifacts."
    )
    args = parser.parse_args()

    run_training(data_path=args.data_path, model_output_path=args.model_output_path)