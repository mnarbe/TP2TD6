import os
import pandas as pd
import numpy as np
import math
import time
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK

pd.set_option("display.max_columns", None)

# Ajustes generales
COMPETITION_PATH = ""
RAND_SEED = 251
RAND_SEED_2 = 328
PORCENTAJE_DATASET_UTILIZADO = 0.3
MAX_EVALS_BAYESIAN = 10
FOLD_SPLITS = 5

def load_competition_datasets(data_dir, sample_frac=None, random_state=None):
    print("Loading competition datasets from:", data_dir)
    train_file = os.path.join(data_dir, "merged_data.csv")
    test_file = os.path.join(data_dir, "test_data.txt")

    train_df = pd.read_csv(train_file, low_memory=False)
    if sample_frac is not None:
        train_df = train_df.sample(frac=sample_frac, random_state=random_state)

    test_df = pd.read_csv(test_file, sep="\t", low_memory=False)
    combined = pd.concat([train_df, test_df], ignore_index=True)
    print(f"  -> Concatenated DataFrame: {combined.shape[0]} rows")
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
    print("Casting column types...")

    # Mantener datetime para feature engineering
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    
    df["offline_timestamp"] = pd.to_datetime(df["offline_timestamp"], errors="coerce", utc=True)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["album_release_date"] = pd.to_datetime(df["album_release_date"], errors="coerce")
    
    # Hago variables que combinen otras:
    # Horario + Tipo de contenido
    df["time_track"] = (df["time_of_day"].astype(str) + "_" + df["is_track"].astype(str)).astype("category")
    df["time_podcast"] = (df["time_of_day"].astype(str) + "_" + df["is_podcast"].astype(str)).astype("category")

    # Usuario + Plataforma
    df["user_os"] = (df["username"].astype(str) + "_" + df["operative_system"].astype(str)).astype("category")
    df["user_shuffle"] = (df["username"].astype(str) + "_" + df["shuffle"].astype(str)).astype("category")

    # País + Dispositivo / Plataforma
    df["country_os"] = (df["conn_country"].astype(str) + "_" + df["operative_system"].astype(str)).astype("category")
    df["country_platform"] = (df["conn_country"].astype(str) + "_" + df["platform"].astype(str)).astype("category")

    # Dispositivo + horario
    df["device_time"] = (df["operative_system"].astype(str) + "_" + df["time_of_day"].astype(str)).astype("category")
    
    # Contenido + Popularidad
    df["artist_popularity"] = (df["artist_name"].astype(str) + "_" + df["popularity"].astype(str)).astype("category")
    df["show_length"] = (df["show_name"].astype(str) + "_" + df["show_total_episodes"].astype(str)).astype("category")


    # Mes + Hora
    df["month_hour"] = (df["month_played"].astype(str) + "_" + df["ts"].dt.hour.astype(str)).astype("category")


    # Columnas categóricas
    cat_cols = [
        "platform", "conn_country", "ip_addr", "master_metadata_track_name",
        "master_metadata_album_artist_name", "master_metadata_album_album_name",
        "episode_name", "episode_show_name", "audiobook_title",
        "audiobook_chapter_title", "name", "album_name", "artist_name",
        "show_name", "show_publisher", "time_track", "time_podcast",
        "user_os", "user_shuffle", "country_os", "country_platform",
        "device_time", "artist_popularity", "show_length", "month_hour"
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # Columnas booleanas
    bool_cols = ["shuffle", "offline", "incognito_mode", "explicit"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype(bool)

    # Columnas numéricas
    int_cols = ["obs_id", "duration_ms", "popularity", "track_number", "show_total_episodes"]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int64)

    print("  -> Column types cast successfully.")
    return df

def split_train_test(X, y, test_mask):
    print("Splitting data into train/test sets...")
    train_mask = ~test_mask
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    print(f"  -> Training set: {X_train.shape[0]} rows")
    print(f"  -> Test set:     {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test

def train_classifier_xgboost_val(X_train, y_train, X_val=None, y_val=None, params=None):
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        seed=RAND_SEED,
        eval_metric='auc',
        enable_categorical=True,
        **params
    )
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train)
    return model

def objective_GroupKFold(params, X_train, y_train, groups):
    gkf = GroupKFold(n_splits=FOLD_SPLITS)
    aucs = []
    for i, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        model = train_classifier_xgboost_val(X_tr, y_tr, X_va, y_va, params)
        preds = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, preds)
        aucs.append(auc)
        print(f"  -> Fold {i+1} AUC: {auc:.5f}")   # <- esto imprime cada fold
    mean_auc = np.mean(aucs)
    print(f"  -> Mean AUC: {mean_auc:.5f}\n")
    return {"loss": 1 - mean_auc, "status": STATUS_OK}


def main():
    start = time.time()
    print("=== Starting pipeline ===")

    df = load_competition_datasets(
        COMPETITION_PATH, sample_frac=PORCENTAJE_DATASET_UTILIZADO, random_state=RAND_SEED
    )
    df = cast_column_types(df)

    # Feature engineering
    df["month_played"] = df["ts"].dt.month.astype("uint8")
    df["time_of_day"] = df["ts"].dt.hour.apply(momento_del_dia).astype("category")
    df["is_track"] = df["master_metadata_track_name"].notna().astype(bool)
    df["is_podcast"] = df["episode_name"].notna().astype(bool)
    df["operative_system"] = df["platform"].str.strip().str.split(n=1).str[0].astype("category")
    df["is_2024"] = (df["ts"].dt.year==2024).astype(bool)
    df = df.sort_values(["obs_id"])

    # Target y test mask
    df["target"] = (df["reason_end"] == "fwdbtn").astype(int)
    df["is_test"] = df["reason_end"].isna()
    df.drop(columns=["reason_end"], inplace=True)

    to_keep = [
        "obs_id","target","is_test","incognito_mode","offline","shuffle","username",
        "conn_country","month_played","time_of_day","is_track","is_podcast",
        "operative_system","episode_name","episode_show_name","audiobook_title",
        "audiobook_chapter_title","offline_timestamp","name","duration_ms","explicit",
        "release_date","album_name","artist_name","popularity","track_number",
        "show_name","show_publisher","show_total_episodes", "is_2024", "time_track", "time_podcast",
        "user_os", "user_shuffle", "country_os", "country_platform",
        "device_time", "artist_popularity", "show_length", "month_hour"
    ]
    df = df[to_keep]

    test_mask = df["is_test"].to_numpy()
    y = df["target"].to_numpy()
    X = df.drop(columns=["target", "is_test"])

    X_train_inicial, X_test, y_train_inicial, y_test = split_train_test(X, y, test_mask)
    test_obs_ids = X_test["obs_id"].copy()

    # Grupos para GroupKFold
    train_groups = X_train_inicial["username"].values

    # Remover identificadores de features
    drop_if_exists = ["obs_id", "username"]
    X_train_inicial = X_train_inicial.drop(columns=[c for c in drop_if_exists if c in X_train_inicial.columns])
    X_test = X_test.drop(columns=[c for c in drop_if_exists if c in X_test.columns])

    # Convertir fechas a int64 antes de XGBoost
    date_cols = ["ts", "offline_timestamp", "release_date", "album_release_date"]
    for col in date_cols:
        if col in X_train_inicial.columns:
            X_train_inicial[col] = X_train_inicial[col].astype("int64")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("int64")

    # Hyperparameter search space
    space = {
        'max_depth': hp.uniformint('max_depth', 3, 30),
        'gamma': hp.uniform('gamma', 0, 4),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
        'reg_lambda': hp.uniform('reg_lambda', 0, 4),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'n_estimators': hp.uniformint('n_estimators', 10, 150),
        'min_child_weight': hp.uniformint('min_child_weight', 1, 10)
    }

    # Bayesian optimization
    best = fmin(
        fn=lambda params: objective_GroupKFold(params, X_train_inicial, pd.Series(y_train_inicial), train_groups),
        space=space,
        algo=tpe.suggest,
        max_evals=MAX_EVALS_BAYESIAN,
        rstate=np.random.default_rng(RAND_SEED_2),
    )
    params = space_eval(space, best)
    print("\nBest hyperparameters found:")
    print(params)

    # Entrenar modelo final
    model = train_classifier_xgboost_val(X_train_inicial, y_train_inicial, None, None, params)

    # Feature importances
    importances = model.feature_importances_
    imp_series = pd.Series(importances, index=model.get_booster().feature_names)
    imp_sorted = imp_series.sort_values(ascending=False)
    print("\nTop 20 feature importances:")
    print(imp_sorted.head(20))

    # Predicciones test
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds_df = pd.DataFrame({"obs_id": test_obs_ids, "pred_proba": preds_proba})
    preds_df.to_csv("modelo_benchmark.csv", index=False)
    print(f"  -> Predictions written to 'modelo_benchmark.csv'")

    print("=== Pipeline complete ===")
    end = time.time()
    print(f'Tiempo transcurrido: {str(end - start)} segundos')

if __name__ == "__main__":
    main()
