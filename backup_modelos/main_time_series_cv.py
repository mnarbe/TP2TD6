# ================= USA TIME SERIES CROSS VALIDATION =================

import pandas as pd
import numpy as np
import time
from database_utils import load_competition_datasets, cast_column_types, split_train_test, processFinalInformation, createNewFeatures, processTargetAndTestMask, keepImportantColumnsDefault
from base_xgboost import trainXGBoostModelTemporal
import constants as C
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)

# Adjust this path if needed
PORCENTAJE_DATASET_UTILIZADO = 1 # Porcentaje del dataset a utilizar
MAX_EVALS_BAYESIAN = 3 # Cantidad de iteraciones para la optimización bayesiana

def objective_time_series_cv(params, X, y):
    """
    Evalúa los hiperparámetros usando validación cruzada temporal (TimeSeriesSplit)
    y devuelve el error (1 - AUC promedio) para que Hyperopt lo minimice.
    """
    auc_scores = []
    print("VALIDACION UNIQUES: ", X["year_ts"].unique())
    print(X["year_ts"].dtype, X["year_ts"].unique())
    for cut in [2024]:
        train_mask = X["year_ts"] < cut
        valid_mask = X["year_ts"] == cut

        X_train_cv = X.loc[train_mask].drop(columns=["year_ts"], errors="ignore")
        y_train_cv = y.loc[train_mask]
        X_valid_cv = X.loc[valid_mask].drop(columns=["year_ts"], errors="ignore")
        y_valid_cv = y.loc[valid_mask]

        # Evitar folds vacíos
        if X_train_cv.empty or X_valid_cv.empty:
            continue

        model = train_classifier_xgboost_val(X_train_cv, y_train_cv, X_valid_cv, y_valid_cv, params)
        y_pred_proba = model.predict_proba(X_valid_cv)[:, 1]
        auc = roc_auc_score(y_valid_cv, y_pred_proba)
        auc_scores.append(auc)

    auc_final = 1 - np.mean(auc_scores)
    print(f"  --> Temporal Validation AUC: {auc_final:.5f}")

    # Hyperopt minimiza la loss, por eso usamos 1 - AUC
    return auc_final

def main(evals):
    start = time.time()
    print("=== Starting pipeline with temporal validation ===")

    if evals > 0:
        max_evals = evals
    else:
        max_evals = MAX_EVALS_BAYESIAN

    # Load and preprocess data
    df = load_competition_datasets(
        C.COMPETITION_PATH, sample_frac=PORCENTAJE_DATASET_UTILIZADO, random_state=C.RAND_SEED
    )
    
    # Preprocess dataset
    df = cast_column_types(df)
    df = createNewFeatures(df)
    df = df.sort_values(["obs_id"])
    df = processTargetAndTestMask(df)
    
    # Split off the actual test set for final predictions
    X_train_dataset, X_test_to_predict, y_train_dataset, _ = split_train_test(df)
    test_obs_ids = X_test_to_predict["obs_id"].copy()

    # Seleccionar las últimas 2 semanas como test
    print("Performing temporal split (last 2 weeks as test)...")
    # X_train_dataset = X_train_dataset.copy()
    # X_train_dataset["ts"] = pd.to_datetime(X_train_dataset["ts"], errors="coerce", utc=True)
    # Ordenar por timestamp
    X_train_dataset = X_train_dataset.sort_values("ts")
    y_train_dataset = y_train_dataset[X_train_dataset.index]

    # Calcular el límite de fecha para las últimas 2 semanas y crear mascaras
    max_date = X_train_dataset["ts"].max()
    test_start = max_date - pd.Timedelta(weeks=2)
    temporal_val_mask = X_train_dataset["ts"] >= test_start
    temporal_train_mask = X_train_dataset["ts"] < test_start

    # Dejar solo las columnas importantes
    X_train_dataset = keepImportantColumnsDefault(X_train_dataset)
    X_test_to_predict = keepImportantColumnsDefault(X_test_to_predict)

    # Splittear datos
    X_train = X_train_dataset[temporal_train_mask].copy()
    X_test = X_train_dataset[temporal_val_mask].copy()
    y_train = y_train_dataset[temporal_train_mask]
    y_test = y_train_dataset[temporal_val_mask]

    print(f"  --> Temporal training set: {X_train.shape[0]} rows")
    print(f"  --> Temporal test set (last 2 weeks): {X_test.shape[0]} rows")
    
    # Remove obs_id and year from feature matrices (keep for final predictions)
    X_train_features = X_train.drop(columns=["obs_id"])
    X_test_features = X_test.drop(columns=["obs_id", "year_ts"])
    X_test_to_predict_features = X_test_to_predict.drop(columns=["obs_id", "year_ts"])

    print(f"Target distribution in training: {y_train.mean():.4f}")
    print(f"Target distribution in testing: {y_test.mean():.4f}")

    # # Train model with temporal validation using the NEW function
    y_train = pd.Series(y_train, index=X_train_features.index)
    model = trainXGBoostModelTemporal(
        X_train_features, y_train, 
        None, None, 
        max_evals, True
    )

    # For final evaluation, use the test set
    processFinalInformation(model, X_test_features, y_test, X_test_to_predict_features, test_obs_ids)

    print("=== Pipeline complete ===")
    end = time.time()
    print(f'Tiempo transcurrido: {str(end - start)} segundos')

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]  # argumentos desde línea de comando
    if args:
        max_evals = int(args[0])
        main(max_evals)
    else:
        main(0)