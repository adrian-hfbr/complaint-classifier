# src/tune.py
import argparse
import pandas as pd
import joblib
import mlflow
from scipy.stats import randint
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .preprocessing import ComplaintPreprocessor
from .utils import get_git_commit_hash


def run_tuning(data_path: str, model_output_path: str):
    """
    Performs hyperparameter tuning using RandomizedSearchCV and logs the best model.
    """
    with mlflow.start_run(): # begins the experiment tracking
        mlflow.set_tag("mlflow.runName", "Hyperparameter Tuning")

        # --- 1. Load Data & Split ---
        df_raw = pd.read_csv(
            data_path,
            nrows=150000, # Use a subsample for faster development
            low_memory=False # Silences the DtypeWarning
        )
        preprocessor = ComplaintPreprocessor()
        df_clean = preprocessor._clean_and_filter(df_raw)
        
        df_train, _ = train_test_split(
            df_clean, test_size=0.2, random_state=42, stratify=df_clean[preprocessor.target_column]
        )

        # --- 2. Create the Pipeline ---
        # chains the preprocessor and the classifier together
        pipeline = Pipeline([
            ('preprocessor', ComplaintPreprocessor()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        # --- 3. Define Hyperparameter Search Space ---
        # defining ranges or distributions for each hyperparameter
        param_dist = {
            'classifier__n_estimators': randint(50, 200),
            'classifier__max_depth': randint(10, 50),
            'classifier__min_samples_split': randint(2, 11),
            'classifier__min_samples_leaf': randint(1, 11),
            'classifier__criterion': ['gini', 'entropy']
        }

        # --- 4. Set Up Randomized Search with Cross-Validation ---
        # For every iteration, split into training & validation set
        # preprocessor-fit on training set, transform training set
        # train RF on training set, evaluate RF on validation set
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=5, # n different hyperparameter combinations are tested
            cv=3, # with 3-fold cv
            verbose=2,
            random_state=42,
            n_jobs=-1 # Use all available CPU cores
        )

        # --- 4. Run the Search ---
        print("Running RandomizedSearchCV...")
        # Trains in total n*k models (n iterations * k CV folds)
        random_search.fit(df_train, df_train[preprocessor.target_column])
        print("Search complete.")

        # --- 5. Log Results and Save Best Model ---
        # Step 5.1: Extract the best results from the search object
        best_pipeline = random_search.best_estimator_
        best_params = random_search.best_params_
        best_score = random_search.best_score_

        # Step 5.2: Print, Log & Save
        print(f"Best Score (CV Accuracy/F1): {best_score:.4f}")
        print("Best Parameters found:")
        for param, value in best_params.items():
            print(f"  - {param}: {value}")
        
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_score", best_score)

        pipeline_path = f"{model_output_path}/best_pipeline.pkl"
        joblib.dump(best_pipeline, pipeline_path)
        mlflow.log_artifact(pipeline_path)
        print(f"Best pipeline saved to {pipeline_path} and logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune a complaint classification model.")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model-output-path", type=str, required=True)
    args = parser.parse_args()
    
    run_tuning(data_path=args.data_path, model_output_path=args.model_output_path)