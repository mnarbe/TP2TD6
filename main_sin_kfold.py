# ================= ARMA XGBOOST SEPARANDO TRAIN=<2024, VAL=2024, TEST=TEST_SET =================

import pandas as pd
import time
from database_utils import load_competition_datasets, cast_column_types, split_train_test, processFinalInformation, createNewFeatures, processTargetAndTestMask, keepImportantColumnsDefault
from base_xgboost import trainXGBoostModelTemporal
import constants as C
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)

# Adjust this path if needed
PORCENTAJE_DATASET_UTILIZADO = 1 # Porcentaje del dataset a utilizar
MAX_EVALS_BAYESIAN = 5 # Cantidad de iteraciones para la optimización bayesiana

def main():
    start = time.time()
    print("=== Starting pipeline with temporal validation ===")

    # Load and preprocess data
    df = load_competition_datasets(
        C.COMPETITION_PATH, sample_frac=PORCENTAJE_DATASET_UTILIZADO, random_state=C.RAND_SEED
    )
    
    # Preprocess dataset
    df = cast_column_types(df)
    df = createNewFeatures(df)
    df = df.sort_values(["obs_id"])
    df = processTargetAndTestMask(df)
    df = keepImportantColumnsDefault(df)
    df.drop(columns=["ts", "spotify_track_uri"]) # No la necesitamos en este main
    
    # Split off the actual test set for final predictions
    X_train_dataset, X_test_to_predict, y_train_dataset, _ = split_train_test(df)
    test_obs_ids = X_test_to_predict["obs_id"].copy()
    
    # Now do temporal split on the training data
    print("Performing temporal split...")
    print(f"Available years in training data: {sorted(X_train_dataset['year_ts'].unique())}")
    
    # Split: pre-2024 for training, 2024 for validation
    temporal_train_mask = X_train_dataset["year_ts"] < 2024
    temporal_val_mask = X_train_dataset["year_ts"] == 2024
    
    X_train = X_train_dataset[temporal_train_mask].copy()
    X_val = X_train_dataset[temporal_val_mask].copy()
    y_train = y_train_dataset[temporal_train_mask]
    y_val = y_train_dataset[temporal_val_mask]

    # Get 10% from 2024 to make a test set
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=0.1, random_state=C.RAND_SEED, stratify=y_val
    )
    
    print(f"  --> Temporal training set (pre-2024): {X_train.shape[0]} rows")
    print(f"  --> Temporal validation set (2024): {X_val.shape[0]} rows")
    print(f"  --> Temporal test set (2024): {X_test.shape[0]} rows")
    
    # Check if we have enough data in both splits
    if X_val.shape[0] == 0:
        print("WARNING: No 2024 data found for validation. Falling back to random split.")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_dataset, y_train_dataset, test_size=0.2, 
            random_state=C.RAND_SEED, stratify=y_train_dataset
        )
    elif X_train.shape[0] == 0:
        print("WARNING: No pre-2024 data found for training. This won't work well.")
        return
    
    # Remove obs_id and year from feature matrices (keep for final predictions)
    X_train_features = X_train.drop(columns=["obs_id", "year_ts"])
    X_val_features = X_val.drop(columns=["obs_id", "year_ts"])
    X_test_features = X_test.drop(columns=["obs_id", "year_ts"])
    X_test_to_predict_features = X_test_to_predict.drop(columns=["obs_id", "year_ts"])
    
    print(f"Target distribution in training: {y_train.mean():.4f}")
    print(f"Target distribution in validation: {y_val.mean():.4f}")

    # Train model with temporal validation using the NEW function
    model = trainXGBoostModelTemporal(
        X_train_features, y_train, 
        X_val_features, y_val, 
        MAX_EVALS_BAYESIAN
    )

    # For final evaluation, use the test set
    processFinalInformation(model, X_test_features, y_test, X_test_to_predict_features, test_obs_ids)

    print("=== Pipeline complete ===")
    end = time.time()
    print(f'Tiempo transcurrido: {str(end - start)} segundos')

if __name__ == "__main__":
    main()