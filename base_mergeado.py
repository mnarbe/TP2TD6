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

# ------------------------ Funciones ------------------------
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

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["offline_timestamp"] = pd.to_datetime(df["offline_timestamp"], errors="coerce", utc=True)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["album_release_date"] = pd.to_datetime(df["album_release_date"], errors="coerce")
    
    # Combinaciones de columnas
    df["time_track"] = (df["time_of_day"].astype(str) + "_" + df["is_track"].astype(str)).astype("category")
    df["time_podcast"] = (df["time_of_day"].astype(str) + "_" + df["is_podcast"].astype(str)).astype("category")
    df["user_os"] = (df["username"].astype(str) + "_" + df["operative_system"].astype(str)).astype("category")
    df["user_shuffle"] = (df["username"].astype(str) + "_" + df["shuffle"].astype(str)).astype("category")
    df["country_os"] = (df["conn_country"].astype(str) + "_" + df["operative_system"].astype(str)).astype("category")
    df["country_platform"] = (df["conn_country"].astype(str) + "_" + df["platform"].astype(str)).astype("category")
    df["device_time"] = (df["operative_system"].astype(str) + "_" + df["time_of_day"].astype(str)).astype("category")
    df["artist_popularity"] = (df["artist_name"].astype(str) + "_" + df["popularity"].astype(str)).astype("category")
    df["show_length"] = (df["show_name"].astype(str) + "_" + df["show_total_episodes"].astype(str)).astype("category")
    df["month_hour"] = (df["month_played"].astype(str) + "_" + df["ts"].dt.hour.astype(str)).astype("category")
    df["time_since_release"] = (df["ts"] - df["release_date"]).dt.total_seconds().fillna(0)

    # Categóricas
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

    # Booleanas
    bool_cols = ["shuffle", "offline", "incognito_mode", "explicit"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype(bool)

    # Numéricas
    int_cols = ["obs_id", "duration_ms", "popularity", "track_number", "show_total_episodes"]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int64)

    print("  -> Column types cast successfully.")
    return df

# Funcion que obtiene features de usuario sin leakage al no mirar observaciones futuras ni del test
def add_user_features(X, y=None):
    X_new = X.copy()

    if y is not None: #si no es test
        X_new = X_new.sort_values(["username", "ts"])
        X_new["user_avg_duration"] = X_new.groupby("username")["duration_ms"].apply(lambda x: x.expanding().mean().shift()) #expanding().mean() se encarga que sean pasadas o actuales, y el shift lo tira una para abajo así solo usa pasadas
        X_new["user_avg_duration"].fillna(X_new["duration_ms"].mean(), inplace=True)
    else:
        X_new["user_avg_duration"] = X_new["duration_ms"].mean()

    X_new["track_longer_than_user_avg"] = (X_new["duration_ms"] > X_new["user_avg_duration"]).astype(int)
    return X_new

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
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
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
        print(f"  -> Fold {i+1} AUC: {auc:.5f}")
    mean_auc = np.mean(aucs)
    print(f"  -> Mean AUC: {mean_auc:.5f}\n")
    return {"loss": 1 - mean_auc, "status": STATUS_OK}

# ------------------------ MAIN ------------------------
def main():
    start = time.time()
    print("=== Starting pipeline ===")

    df = load_competition_datasets(COMPETITION_PATH, sample_frac=PORCENTAJE_DATASET_UTILIZADO, random_state=RAND_SEED)
    df = cast_column_types(df)

    df["month_played"] = df["ts"].dt.month.astype("uint8")
    df["time_of_day"] = df["ts"].dt.hour.apply(momento_del_dia).astype("category")
    df["is_track"] = df["master_metadata_track_name"].notna().astype(bool)
    df["is_podcast"] = df["episode_name"].notna().astype(bool)
    df["operative_system"] = df["platform"].str.strip().str.split(n=1).str[0].astype("category")
    df["is_2024"] = (df["ts"].dt.year == 2024).astype(bool)
    df = df.sort_values(["obs_id"])

    df["target"] = (df["reason_end"] == "fwdbtn").astype(int)
    df["is_test"] = df["reason_end"].isna()
    df.drop(columns=["reason_end"], inplace=True)

    to_keep = [
        "obs_id","target","is_test","incognito_mode","offline","shuffle","username",
        "conn_country","month_played","time_of_day","is_track","is_podcast",
        "operative_system","episode_name","episode_show_name","audiobook_title",
        "audiobook_chapter_title","offline_timestamp","name","duration_ms","explicit",
        "release_date","album_name","artist_name","popularity","track_number",
        "show_name","show_publisher","show_total_episodes","is_2024","time_track","time_podcast",
        "user_os","user_shuffle","country_os","country_platform","device_time","artist_popularity","show_length","month_hour"
    ]
    df = df[to_keep]

    test_mask = df["is_test"].to_numpy()
    y = df["target"].to_numpy()
    X = df.drop(columns=["target", "is_test"])

    X_train_inicial, X_test, y_train_inicial, y_test = split_train_test(X, y, test_mask)
    test_obs_ids = X_test["obs_id"].copy()

    X_train_inicial = add_user_features(X_train_inicial, y_train_inicial)
    X_test = add_user_features(X_test)

    train_groups = X_train_inicial["username"].values

    drop_if_exists = ["obs_id", "username"]
    X_train_inicial = X_train_inicial.drop(columns=[c for c in drop_if_exists if c in X_train_inicial.columns])
    X_test = X_test.drop(columns=[c for c in drop_if_exists if c in X_test.columns])

    date_cols = ["ts", "offline_timestamp", "release_date", "album_release_date"]
    for col in date_cols:
        if col in X_train_inicial.columns:
            X_train_inicial[col] = X_train_inicial[col].astype("int64")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("int64")

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

    model = train_classifier_xgboost_val(X_train_inicial, y_train_inicial, None, None, params)

    importances = model.feature_importances_
    imp_series = pd.Series(importances, index=model.get_booster().feature_names)
    imp_sorted = imp_series.sort_values(ascending=False)
    print("\nTop 20 feature importances:")
    print(imp_sorted.head(20))

    preds_proba = model.predict_proba(X_test)[:, 1]
    preds_df = pd.DataFrame({"obs_id": test_obs_ids, "pred_proba": preds_proba})
    preds_df.to_csv("modelo_benchmark.csv", index=False)
    print(f"  -> Predictions written to 'modelo_benchmark.csv'")

    train_preds = model.predict_proba(X_train_inicial)[:, 1]
    train_auc = roc_auc_score(y_train_inicial, train_preds)
    print(f"\nFinal training AUC: {train_auc:.5f}")

    print("=== Pipeline complete ===")
    end = time.time()
    print(f'Tiempo transcurrido: {str(end - start)} segundos')

if __name__ == "__main__":
    main()
