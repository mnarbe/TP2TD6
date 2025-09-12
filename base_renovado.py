import os
import pandas as pd
import numpy as np
import math
import time
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK

pd.set_option("display.max_columns", None)

# Adjust this path if needed
COMPETITION_PATH = ""
RAND_SEED = 251
RAND_SEED_2 = 328
PORCENTAJE_DATASET_UTILIZADO = 0.3 # Porcentaje del dataset a utilizar (0.0-1.0)
MAX_EVALS_BAYESIAN = 10 # Cantidad de iteraciones para la optimización bayesiana
FOLD_SPLITS = 3 # Cantidad de folds (KFold o GroupKFold)

def load_competition_datasets(data_dir, sample_frac=None, random_state=None):
    """
    Load train and test datasets, optionally sample a fraction of the training set,
    concatenate, and reset index.
    """
    print("Loading competition datasets from:", data_dir)
    train_file = os.path.join(data_dir, "train_data.txt")
    test_file = os.path.join(data_dir, "test_data.txt")

    # Load training data and optionally subsample
    train_df = pd.read_csv(train_file, sep="\t", low_memory=False)
    if sample_frac is not None:
        train_df = train_df.sample(frac=sample_frac, random_state=random_state)

    # Load test data
    test_df = pd.read_csv(test_file, sep="\t", low_memory=False)

    # Concatenate and reset index
    combined = pd.concat([train_df, test_df], ignore_index=True)
    print(f"  → Concatenated DataFrame: {combined.shape[0]} rows")
    return combined

def momento_del_dia(hora):
    if 6 <= hora < 10:
        return "morning"
    elif 10 <= hora < 14:
        return "noon"
    elif 14 <= hora < 18:
        return "afternoon"
    elif 18 <= hora < 22:
        return "evening"
    elif 22 <= hora < 24:
        return "late_night"
    else:
        return "early_morning"


def cast_column_types(df):
    """
    Cast columns to efficient dtypes and parse datetime fields.
    """
    print("Casting column types and parsing datetime fields...")
    dtype_map = {
        #"platform": "category",
        "conn_country": "category",
        "ip_addr": "category",
        "master_metadata_track_name": "category",
        "master_metadata_album_artist_name": "category",
        "master_metadata_album_album_name": "category",
        "reason_end": "category",
        "username": "category",
        "spotify_track_uri": "string",
        "episode_name": "category",
        "episode_show_name": "category",
        "spotify_episode_uri": "string",
        "audiobook_title": "string",
        "audiobook_uri": "string",
        "audiobook_chapter_uri": "string",
        "audiobook_chapter_title": "string",
        "shuffle": bool,
        "offline": bool,
        "incognito_mode": bool,
        "obs_id": int
    }

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["offline_timestamp"] = pd.to_datetime(
        df["offline_timestamp"], unit="s", errors="coerce", utc=True
    )
    df = df.astype(dtype_map)
    print("  → Column types cast successfully.")
    return df


def split_train_test(X, y, test_mask):
    """
    Split features and labels into train/test based on mask.
    """
    print("Splitting data into train/test sets...")

    train_mask = ~test_mask  # Invertir la máscara

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    print(f"  → Training set: {X_train.shape[0]} rows")
    print(f"  → Test set:     {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test


def train_classifier_basic(X_train, y_train, params=None):
    """
    Train a Classifier 
    """
    print("Training model...")
    
    default_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": RAND_SEED,
        "bootstrap": True,
    }
    rf_params = default_params.copy()
    if params:
        rf_params.update(params)

    model = RandomForestClassifier(**rf_params)

    print("  → Fitting RandomForestClassifier...")
    model.fit(X_train, y_train)
    print("  → Model training complete.")
    return model

def train_classifier_xgboost(X_train, y_train, params=None):
    """
    Train a Classifier 
    """

    model = xgb.XGBClassifier(objective = 'binary:logistic',
                                seed = RAND_SEED,
                                eval_metric = 'auc',
                                enable_categorical=True,
                                **params)

    model.fit(X_train, y_train)
    return model

def train_classifier_xgboost_val(X_train, y_train, X_val, y_val, params=None):
    """
    Train a Classifier 
    """

    model = xgb.XGBClassifier(objective = 'binary:logistic',
                                seed = RAND_SEED,
                                eval_metric = 'auc',
                                enable_categorical=True,
                                **params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)]
    )
    return model

def objective_KFold(params, X_train, y_train):
    """
    CV interna sobre el train inicial (sin agrupar por usuario).
    """
    kf = KFold(n_splits=FOLD_SPLITS, shuffle=True, random_state=RAND_SEED)
    aucs = []
    for tr_idx, va_idx in kf.split(X_train, y_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        model = train_classifier_xgboost_val(X_tr, y_tr, X_va, y_va, params)
        preds = model.predict_proba(X_va)[:, 1]
        aucs.append(roc_auc_score(y_va, preds))
    return {"loss": 1 - np.mean(aucs), "status": STATUS_OK}


def main():
    start = time.time()
    print("=== Starting pipeline ===")

    # Load and preprocess data
    df = load_competition_datasets(
        COMPETITION_PATH, sample_frac=PORCENTAJE_DATASET_UTILIZADO, random_state=RAND_SEED
    )
    
    df = cast_column_types(df)
    #Agrego mes
    df["month_played"] = df["ts"].dt.month.astype("uint8")

    #Agrego hora --> rango horario: madrugada, mañana, mediodia, tarde, noche, muy noche
    df["time_of_day"] = df["ts"].dt.hour.apply(momento_del_dia).astype("category")
    
    #Agrego flag podcast vs track: podemos hacerla una sola flag que sea true si es cancion, false si es podcast. La dejo así por si acaso por si genera un error unirlas.
    df["is_track"] = df["master_metadata_track_name"].notna().astype("uint8")
    df["is_podcast"] = df["episode_name"].notna().astype("uint8")
    
    #Hago que solo lea la primera palabra de platform (así no separa cada windows, por ejemplo)
    df["operative_system"] = df["platform"].str.strip().str.split(n=1).str[0].astype("category")
    
    # df["user_order"] = df.groupby("username", observed=True).cumcount() + 1
    df = df.sort_values(["obs_id"])

    # Create target and test mask
    print("Creating 'target' and 'is_test' columns...")
    df["target"] = (df["reason_end"] == "fwdbtn").astype(int)
    df["is_test"] = df["reason_end"].isna()
    df.drop(columns=["reason_end"], inplace=True)
    print("  → 'target' and 'is_test' created, dropped 'reason_end' column.")

    # Keep only relevant columns
    to_keep = [
        "obs_id",
        "target",
        "is_test",
        "incognito_mode",
        "offline",
        "shuffle",
        "username",
        "conn_country",
        "ip_addr",
        "master_metadata_album_artist_name",
        "master_metadata_track_name", #droppear porque ya tenemos la flag?
        "episode_name", #droppear porque ya tenemos la flag?
        #"ts",
        "month_played",
        "time_of_day",
        "is_track",
        "is_podcast",
        "operative_system"
    ]
    
    df = df[to_keep]

    # Define hyperparameter search space
    space = {
        'max_depth': hp.uniformint('max_depth', 3, 30),
        'gamma': hp.uniform('gamma', 0, 4),                    # Regularización, suele estar entre 0 y 4
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2), # Típico entre 0.01 y 0.2
        'reg_lambda': hp.uniform('reg_lambda', 0, 4),           # Regularización L2, 0 a 4
        'reg_alpha': hp.uniform('reg_alpha', 0, 4),             # Regularización L1, 0 a 4
        'subsample': hp.uniform('subsample', 0.5, 1),           # Fracción de muestras, 0.5 a 1
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1), # Fracción de columnas, 0.5 a 1
        'n_estimators': hp.uniformint('n_estimators', 10, 150), # Número de árboles, 10 a 150
        'min_child_weight': hp.uniform('min_child_weight', 1, 10) # Peso mínimo de hijos, 1 a 10
    }

    # Build feature matrix and get feature names
    test_mask = df["is_test"].to_numpy()
    y = df["target"].to_numpy()
    X = df.drop(columns=["target", "is_test"])

    # Split data
    X_train_inicial, X_test, y_train_inicial, _ = split_train_test(X, y, test_mask)

    # Guardar obs_id
    test_obs_ids = X_test["obs_id"].copy()
    
    X_train = None
    y_train = None
    X_val = None
    y_val = None
    best = None

    X = X.drop(columns=["obs_id"])

    # Split train vs validation normal
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_inicial, y_train_inicial, test_size=0.2, random_state=RAND_SEED, stratify=y_train_inicial
    )

    # Optimización de hiperparametros
    best = fmin(
        fn=lambda params: objective_KFold(params, X_train, pd.Series(y_train)),
        space=space,
        algo=tpe.suggest,
        max_evals=MAX_EVALS_BAYESIAN,
        rstate=np.random.default_rng(RAND_SEED_2),
    )
    
    params = space_eval(space, best) # Guardamos los hiperparámetros ganadores.
    print("\nBest hyperparameters found:")
    print(params)

    # Train model
    model = train_classifier_xgboost_val(X_train, y_train, X_val, y_val, params)

    # Display top 20 feature importances
    print("\nExtracting and sorting feature importances...")
    importances = model.feature_importances_
    imp_series = pd.Series(importances, index=model.get_booster().feature_names)
    imp_series = imp_series.drop(labels=["obs_id"], errors="ignore")
    imp_sorted = imp_series.sort_values(ascending=False)
    print("\nTop 20 feature importances:")
    print(imp_sorted.head(20))

    # Predict on validation
    print("\nGenerating predictions for validation set...")
    preds_val = model.predict_proba(X_val)[:, 1]
    val_score = roc_auc_score(y_val, preds_val)
    print(f"\nValidation ROC AUC: {val_score}")

    # Predict on test set
    print("\nGenerating predictions for test set...")
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds_df = pd.DataFrame({"obs_id": test_obs_ids, "pred_proba": preds_proba})
    preds_df.to_csv("modelo_benchmark.csv", index=False, sep=",")
    print(f"  → Predictions written to 'modelo_benchmark.csv'")

    print("=== Pipeline complete ===")
    end = time.time()
    print(f'Tiempo transcurrido: {str(end - start)} segundos')


if __name__ == "__main__":
    main()
