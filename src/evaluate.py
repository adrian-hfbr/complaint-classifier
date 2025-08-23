# src/evaluate.py
import argparse
import pandas as pd
import joblib
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from .preprocessing import ComplaintPreprocessor

def run_evaluation(data_path: str, pipeline_path: str):
    """
    Loads the best pipeline, evaluates it on the held-out test set,
    and logs detailed metrics and artifacts to MLflow.
    """
    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", "Final Pipeline Evaluation")
        print("▶️ Starting final pipeline evaluation...")

        # --- 1. Load Data and Clean ---
        df_raw = pd.read_csv(
            data_path, nrows=150000, low_memory=False
        )
        
        # Load the pipeline first to get access to the fitted preprocessor's logic
        pipeline = joblib.load(pipeline_path)
        
        # Use the preprocessor step from the loaded pipeline to clean the raw data
        # before splitting. This ensures the exact same cleaning logic is used.
        preprocessor = pipeline.named_steps['preprocessor']
        df_clean = preprocessor._clean_and_filter(df_raw)

        # --- 2. Create the Held-Out Test Set from CLEANED Data ---
        _, df_test = train_test_split(
            df_clean, # Split the cleaned data
            test_size=0.2,
            random_state=42,
            stratify=df_clean[preprocessor.target_column] # Stratify on the cleaned labels
        )
        y_test = df_test[preprocessor.target_column]

        # --- 3. Predict on Test Data ---
        # The pipeline handles both preprocessing and prediction in one step.
        # Passing the raw text from our test split.
        predictions = pipeline.predict(df_test)

        # --- 4. Generate and Log Evaluation Artifacts ---
        print("\n--- Evaluation Results ---")
        
        # Gets the class labels from the final classifier step of the pipeline
        class_labels = pipeline.named_steps['classifier'].classes_
        
        # Classification Report (logged as a text file)
        report = classification_report(y_test, predictions, target_names=class_labels,
                                       zero_division=0)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")
        print("\nClassification Report:")
        print(report)

        # Confusion Matrix (logged as an image file)
        cm = confusion_matrix(y_test, predictions, labels=class_labels)
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels,
                    yticklabels=class_labels, cmap='Blues')
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('Actual Product', fontsize=12)
        plt.xlabel('Predicted Product', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        print(f"\nConfusion matrix and classification report saved and logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the final model pipeline.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the raw data file.")
    parser.add_argument("--pipeline-path", type=str, required=True, help="Path to the best_pipeline.pkl file.")
    args = parser.parse_args()

    run_evaluation(
        data_path=args.data_path,
        pipeline_path=args.pipeline_path
    )