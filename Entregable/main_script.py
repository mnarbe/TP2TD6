import pandas as pd
import time
from database_utils import *
from base_xgboost import trainXGBoostModelTemporal, backward_feature_selection_topN
import constants as C
import os

pd.set_option("display.max_columns", None)

PORCENTAJE_DATASET_UTILIZADO = 1 # Porcentaje del dataset a utilizar
MAX_EVALS_BAYESIAN = 1 # Cantidad de iteraciones para la optimización bayesiana
MIN_FEATURES = 18 # Cantidad de features mínima que deja backward selection
TOP_N_USED = 1 # Subconjuntos de features a entrenar luego de hacer backward selection (los mejores N)

def runPipeline():
    start = time.time()
    print("=== Starting pipeline with temporal validation ===")

    # ===========================
    # 1. Cargar y preprocesar data
    # ===========================
    df = load_competition_datasets(
        C.COMPETITION_PATH, sample_frac=PORCENTAJE_DATASET_UTILIZADO, random_state=C.RAND_SEED
    )
    df = cast_column_types(df)
    df = createNewFeatures(df) # Crear nuevas features básicas
    df = df.sort_values(["obs_id"])
    df = processTargetAndTestMask(df)
    
    # ===========================
    # 2. Creo counting features (primero spliteo el dataset temporalmente para obtener train set)
    # ===========================
    
    # Separo el train del valid y del test
    df_train_dataset, _ = split_train_test_df(df)
    temporal_train_mask = df_train_dataset["year_ts"] < 2024
    df_train = df_train_dataset[temporal_train_mask].copy()

    # Creo counting features para todo el dataset (No hay leakage porque se ordena temporalmente y los bins se calculan con el train set)
    df = createNewCountingFeatures(df, df_train)

    # ===========================
    # 3. Con el nuevo dataset, spliteo test y valid
    # ===========================

    # Vuelvo a splitear el dataset con las nuevas columnas de counting
    df_train_dataset, X_test_to_predict = split_train_test_df(df)
    X_test_to_predict = X_test_to_predict.drop(columns=["target", "is_test"])

    # Vuelvo a splitear train y valid
    print("Performing temporal split...")
    temporal_train_mask = df_train_dataset["year_ts"] < 2024
    temporal_val_mask = df_train_dataset["year_ts"] == 2024
    df_train = df_train_dataset[temporal_train_mask].copy()
    df_val = df_train_dataset[temporal_val_mask].copy()

    # Ordenar para respetar la secuencia temporal
    df_train = df_train.sort_values(['username', 'ts'])
    df_val = df_val.sort_values(['username', 'ts'])
    X_test_to_predict = X_test_to_predict.sort_values(['username', 'ts'])

    test_obs_ids = X_test_to_predict["obs_id"].copy()

    # ===================================================
    # 4. Split final dentro de 2024 (val/test)
    # ===================================================
    print("final split inside 2024 for validation and test...")

    val_cutoff = df_val["ts"].quantile(0.9)  # 90% early 2024 as validation
    df_val_real = df_val[df_val["ts"] <= val_cutoff]
    df_test = df_val[df_val["ts"] > val_cutoff]

    print(f"  --> final training set (pre-2024): {df_train.shape[0]} rows")
    print(f"  --> final validation set (early 2024): {df_val_real.shape[0]} rows")
    print(f"  --> final test set (late 2024): {df_test.shape[0]} rows")

    # ===================================================
    # 5. Crear features históricas SOLO en train (sin leakage)
    # ===================================================
    print("Creating new features for train set...")
    df_train = createNewSetFeatures(df_train)

    # ===================================================
    # 6. Aplicar features históricas a valid/test
    # ===================================================
    print("Applying new features for valid and test set...")
    df_val_real = applyHistoricalFeaturesToSet(df_val, df_train)
    df_test = applyHistoricalFeaturesToSet(df_test, df_train)
    X_test_to_predict = applyHistoricalFeaturesToSet(X_test_to_predict, df_train)

    # ===================================================
    # 7. Filtrar las columnas importantes (Eliminamos las que no nos resultaron performantes)
    # ===================================================
    
    df_train = keepImportantColumnsDefault(df_train)
    df_val_real = keepImportantColumnsDefault(df_val)
    df_test = keepImportantColumnsDefault(df_test)
    X_test_to_predict = keepImportantColumnsDefault(X_test_to_predict)
    
    # ===================================================
    # 8. Separar X e y
    # ===================================================
    X_train, y_train = split_x_and_y(df_train)
    X_val, y_val = split_x_and_y(df_val_real)
    X_test, y_test = split_x_and_y(df_test)
    
    print(f"  --> Target distribution in training: {y_train.mean():.4f}")
    print(f"  --> Target distribution in validation: {y_val.mean():.4f}")
    print(f"  --> Target distribution in test: {y_test.mean():.4f}")

    # ===================================================
    # 9. Drop columnas auxiliares
    # ===================================================
    # Columnas base a eliminar
    base_cols_to_drop = ["obs_id", "year_ts", "ts", "spotify_track_uri", "date", "hour"]
    
    X_train_features = X_train.drop(columns=base_cols_to_drop, errors='ignore')
    X_val_features = X_val.drop(columns=base_cols_to_drop, errors='ignore')
    X_test_features = X_test.drop(columns=base_cols_to_drop, errors='ignore')
    X_test_to_predict_features = X_test_to_predict.drop(columns=base_cols_to_drop, errors='ignore')

    # ===================================================
    # 10. Aplicar K-Means
    # ===================================================
    # SOLO entrenar K-means en train
    print(f"Entrenando K-Means en train set...")
    X_train_features, kmeans_model = simple_clustering(X_train_features, n_clusters=3)

    # Aplicar el MISMO modelo a validation y test
    print(f"Agrego K-Means model a valid y test...")
    X_val_features, _ = simple_clustering(X_val_features, kmeans_model=kmeans_model)
    X_test_features, _ = simple_clustering(X_test_features, kmeans_model=kmeans_model)
    X_test_to_predict_features, _ = simple_clustering(X_test_to_predict_features, kmeans_model=kmeans_model)

    # ===================================================
    # 11. Aplicar Feature Selection
    # ===================================================

    print(f"Aplico feature selection y obtengo los mejores subconjuntos de features...")
    selected_features_list = backward_feature_selection_topN(
        X_train_features, y_train,
        X_val_features, y_val,
        min_features=MIN_FEATURES,
        topN_used=TOP_N_USED,
        xgb_params=None,
        early_stopping_rounds=50,
        verbose=True
    )

    # Para cada set de features obtenido del backward selection
    for i, selected_features in enumerate(selected_features_list, start=1):
        X_train_selected = X_train_features[selected_features].copy()
        X_val_selected = X_val_features[selected_features].copy()
        X_test_selected = X_test_features[selected_features].copy()
        X_test_to_predict_selected = X_test_to_predict_features[selected_features].copy()

        # ===================================================
        # 12. Entrenar el modelo
        # ===================================================

        model = trainXGBoostModelTemporal(
            X_train_selected, y_train, 
            X_val_selected, y_val, 
            MAX_EVALS_BAYESIAN
        )

        # ===================================================
        # 13. Evaluar y generar predicciones finales
        # ===================================================

        processFinalInformation(model, X_test_selected, y_test, X_test_to_predict_selected, test_obs_ids, best_params=model.get_params())

    print("=== Pipeline complete ===")
    end = time.time()
    print(f'Tiempo transcurrido: {str(end - start)} segundos')

def runAnalisis():
    print("Comenzando Sección de Análisis...")

def main():
    runAnalisis()
    runPipeline()

if __name__ == "__main__":
    main()