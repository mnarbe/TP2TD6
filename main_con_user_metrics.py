import pandas as pd
import time
from database_utils import load_competition_datasets, cast_column_types, split_train_test_df,split_x_and_y, processFinalInformation, createNewFeatures, applyHistoricalFeaturesToSet, processTargetAndTestMask, keepImportantColumnsDefault, createNewSetFeatures, simple_clustering
from base_xgboost import trainXGBoostModelTemporal
import constants as C

pd.set_option("display.max_columns", None)

# Adjust this path if needed
PORCENTAJE_DATASET_UTILIZADO = 1 # Porcentaje del dataset a utilizar
MAX_EVALS_BAYESIAN = 7 # Cantidad de iteraciones para la optimización bayesiana

def main():
    start = time.time()
    print("=== Starting pipeline with temporal validation ===")

    # ===========================
    # 1. Load and preprocess data
    # ===========================
    df = load_competition_datasets(
        C.COMPETITION_PATH, sample_frac=PORCENTAJE_DATASET_UTILIZADO, random_state=C.RAND_SEED
    )
    df = cast_column_types(df)
    df = createNewFeatures(df)
    df = df.sort_values(["obs_id"])
    df = processTargetAndTestMask(df)
    df = keepImportantColumnsDefault(df)
    
    # ===========================
    # 2. Train/Test split (con is_test)
    # ===========================
    df_train_dataset, X_test_to_predict = split_train_test_df(df)
    X_test_to_predict = X_test_to_predict.drop(columns=["target", "is_test"])
    test_obs_ids = X_test_to_predict["obs_id"].copy()
    
    # ===========================
    # 3. Temporal split en train
    # ===========================
    print("Performing temporal split...")
    print(f"Available years in training data: {sorted(df_train_dataset['year_ts'].unique())}")
    
    temporal_train_mask = df_train_dataset["year_ts"] < 2024
    temporal_val_mask = df_train_dataset["year_ts"] == 2024
    df_train = df_train_dataset[temporal_train_mask].copy()
    df_val = df_train_dataset[temporal_val_mask].copy()

    # Ordenar para respetar la secuencia temporal
    df_train = df_train.sort_values(['username', 'ts'])
    df_val = df_val.sort_values(['username', 'ts'])

    # ===================================================
    # 4. Crear features históricas SOLO en train (sin leakage)
    # ===================================================
    print("Creating new features for train set...")
    df_train = createNewSetFeatures(df_train)

    # ===================================================
    # 5. Aplicar features históricas a valid/test
    # ===================================================
    print("Applying new features for valid and test set...")
    df_val = applyHistoricalFeaturesToSet(df_val, df_train)
    X_test_to_predict = applyHistoricalFeaturesToSet(X_test_to_predict, df_train)

    # ===================================================
    # 6. Split temporal dentro de 2024 (val/test)
    # ===================================================
    print("Temporal split inside 2024 for validation and test...")

    val_cutoff = df_val["ts"].quantile(0.9)  # 90% early 2024 as validation
    df_val_real = df_val[df_val["ts"] <= val_cutoff]
    df_test = df_val[df_val["ts"] > val_cutoff]

    print(f"  --> Temporal training set (pre-2024): {df_train.shape[0]} rows")
    print(f"  --> Temporal validation set (early 2024): {df_val_real.shape[0]} rows")
    print(f"  --> Temporal test set (late 2024): {df_test.shape[0]} rows")

    # ===================================================
    # 7. Separar X e y
    # ===================================================
    X_train, y_train = split_x_and_y(df_train)
    X_val, y_val = split_x_and_y(df_val_real)
    X_test, y_test = split_x_and_y(df_test)
    
    # ===================================================
    # 8. Drop columnas auxiliares
    # ===================================================
    X_train_features = X_train.drop(columns=["obs_id", "year_ts", "ts", "spotify_track_uri"])
    X_val_features = X_val.drop(columns=["obs_id", "year_ts", "ts", "spotify_track_uri"])
    X_test_features = X_test.drop(columns=["obs_id", "year_ts", "ts", "spotify_track_uri"])
    X_test_to_predict_features = X_test_to_predict.drop(columns=["obs_id", "year_ts", "ts", "spotify_track_uri"])

    # ===================================================
    # 9. Entrenar modelo
    # ===================================================
    print(f"Target distribution in training: {y_train.mean():.4f}")
    print(f"Target distribution in validation: {y_val.mean():.4f}")


    # SOLO entrenar K-means en train
    X_train_features, kmeans_model = simple_clustering(X_train_features, n_clusters=3)

    # Aplicar el MISMO modelo a validation
    X_val_features, _ = simple_clustering(X_val_features, kmeans_model=kmeans_model)
    
    # Train model with temporal validation using the NEW function
    model = trainXGBoostModelTemporal(
        X_train_features, y_train, 
        X_val_features, y_val, 
        MAX_EVALS_BAYESIAN
    )

    # ===================================================
    # 10. Evaluar y generar predicciones finales
    # ===================================================

    X_test_features, _ = simple_clustering(X_test_features, kmeans_model=kmeans_model)

    processFinalInformation(model, X_test_features, y_test, X_test_to_predict_features, test_obs_ids)

    print("=== Pipeline complete ===")
    end = time.time()
    print(f'Tiempo transcurrido: {str(end - start)} segundos')

if __name__ == "__main__":
    main()