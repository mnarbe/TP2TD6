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


COMPETITION_PATH = ""
RAND_SEED = 251
RAND_SEED_2 = 328
PORCENTAJE_DATASET_UTILIZADO = 0.3 # Porcentaje del dataset a utilizar (0.0-1.0)
MAX_EVALS_BAYESIAN = 10 # Cantidad de iteraciones para la optimización bayesiana
FOLD_SPLITS = 5 # Cantidad de folds (KFold o GroupKFold)

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
    print("  -> Column types cast successfully.")
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

    print(f"  -> Training set: {X_train.shape[0]} rows")
    print(f"  -> Test set:     {X_test.shape[0]} rows")
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

    print("  -> Fitting RandomForestClassifier...")
    model.fit(X_train, y_train)
    print("  -> Model training complete.")
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
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        
        # Verificar que no hay overlap entre train y validation
        assert len(set(tr_idx) & set(va_idx)) == 0, "Overlap between train and validation sets!"
        
        model = train_classifier_xgboost_val(X_tr, y_tr, X_va, y_va, params)
        preds = model.predict_proba(X_va)[:, 1]
        fold_auc = roc_auc_score(y_va, preds)
        aucs.append(fold_auc)
        
        print(f"  Fold {fold+1}/{FOLD_SPLITS}: AUC = {fold_auc:.4f}")
    
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"  CV Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    return {"loss": 1 - mean_auc, "status": STATUS_OK}

def create_user_features(df, train_mask):
    """
    Create user-level features using ONLY training data to avoid data leakage.
    """
    print("Creating user-level features (using only training data)...")
    
    # Use only training data for user statistics
    train_df = df[train_mask].copy()
    
    # User behavior statistics
    user_stats = (
        train_df.groupby("username", observed=True)
        .agg(
            user_total_plays=("obs_id", "count"),
            user_skip_rate=("target", "mean"),
            user_track_rate=("is_track", "mean"),
            user_podcast_rate=("is_podcast", "mean"),
            user_audiobook_rate=("is_audiobook", "mean"),
            user_avg_time_between_plays=("time_since_last_play", "mean"),
            user_offline_rate=("offline", "mean"),
            user_shuffle_rate=("shuffle", "mean"),
            user_incognito_rate=("incognito_mode", "mean"),
            user_nunique_artists=("master_metadata_album_artist_name", pd.Series.nunique),
            user_nunique_countries=("conn_country", pd.Series.nunique),
            user_nunique_ips=("ip_addr", pd.Series.nunique),
        )
    )
    
    # Fill missing values
    user_stats = user_stats.fillna(0)
    
    # Time-of-day preferences
    tod_stats = (
        train_df.groupby(["username", "time_of_day"], observed=True)["obs_id"]
        .count()
        .unstack(fill_value=0)
    )
    tod_totals = tod_stats.sum(axis=1)
    tod_proportions = tod_stats.div(tod_totals, axis=0).fillna(0)
    tod_proportions.columns = [f"user_tod_share_{col}" for col in tod_proportions.columns]
    
    # Combine all user features
    user_features = user_stats.join(tod_proportions, how="left").fillna(0)
    
    # Merge back to full dataset
    df_with_user_features = df.merge(user_features, left_on="username", right_index=True, how="left")
    
    # Fill missing values for users not in training set
    user_feature_cols = user_features.columns
    df_with_user_features[user_feature_cols] = df_with_user_features[user_feature_cols].fillna(0)
    
    print(f"  -> Created {len(user_feature_cols)} user-level features")
    return df_with_user_features, user_feature_cols.tolist()

def objective_GroupKFold(params, X_train, y_train, groups):
    """
    CV interna agrupando por usuario para evitar data leakage.
    """
    gkf = GroupKFold(n_splits=FOLD_SPLITS)
    aucs = []
    
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        
        # Verificar que no hay overlap entre train y validation
        assert len(set(tr_idx) & set(va_idx)) == 0, "Overlap between train and validation sets!"
        
        # Verificar que no hay usuarios compartidos entre train y validation
        train_users = set(groups[tr_idx])
        val_users = set(groups[va_idx])
        assert len(train_users & val_users) == 0, "Users appear in both train and validation!"
        
        model = train_classifier_xgboost_val(X_tr, y_tr, X_va, y_va, params)
        preds = model.predict_proba(X_va)[:, 1]
        fold_auc = roc_auc_score(y_va, preds)
        aucs.append(fold_auc)
        
        print(f"  Fold {fold+1}/{FOLD_SPLITS}: AUC = {fold_auc:.4f} (Train users: {len(train_users)}, Val users: {len(val_users)})")
    
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"  GroupCV Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    return {"loss": 1 - mean_auc, "status": STATUS_OK}


def main():
    start = time.time()
    print("=== Starting pipeline ===")

    # Load and preprocess data
    df = load_competition_datasets(
        COMPETITION_PATH, sample_frac=PORCENTAJE_DATASET_UTILIZADO, random_state=RAND_SEED
    )
    
    df = cast_column_types(df)
    
    # Create target and test mask FIRST to avoid data leakage
    print("Creating 'target' and 'is_test' columns...")
    df["target"] = (df["reason_end"] == "fwdbtn").astype(int)
    df["is_test"] = df["reason_end"].isna()
    df.drop(columns=["reason_end"], inplace=True)
    print("  -> 'target' and 'is_test' created, dropped 'reason_end' column.")
    
    # Sort by obs_id to maintain order
    df = df.sort_values(["obs_id"])
    
    # ========= FEATURE ENGINEERING =========
    print("Creating features...")
    
    # Temporal features
    df["month_played"] = df["ts"].dt.month.astype("uint8")
    df["day_of_week"] = df["ts"].dt.dayofweek.astype("uint8")
    df["hour"] = df["ts"].dt.hour.astype("uint8")
    df["time_of_day"] = df["ts"].dt.hour.apply(momento_del_dia).astype("category")
    
    # Content type flags
    df["is_track"] = df["master_metadata_track_name"].notna().astype("uint8")
    df["is_podcast"] = df["episode_name"].notna().astype("uint8")
    df["is_audiobook"] = df["audiobook_title"].notna().astype("uint8")
    
    # Platform features
    df["operative_system"] = df["platform"].str.strip().str.split(n=1).str[0].astype("category")
    
    # User session features (temporal within user)
    df = df.sort_values(["username", "ts"])
    df["user_order"] = df.groupby("username", observed=True).cumcount() + 1
    df["time_since_last_play"] = df.groupby("username", observed=True)["ts"].diff().dt.total_seconds().fillna(0)
    df["is_first_play_of_day"] = (df.groupby(["username", df["ts"].dt.date], observed=True)["user_order"].rank() == 1).astype("uint8")
    
    # Sort back by obs_id
    df = df.sort_values(["obs_id"])

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
        "master_metadata_track_name",
        "episode_name",
        "month_played",
        "day_of_week",
        "hour",
        "time_of_day",
        "is_track",
        "is_podcast",
        "is_audiobook",
        "operative_system",
        "user_order",
        "time_since_last_play",
        "is_first_play_of_day"
    ]
    
    df = df[to_keep]

    # Create train mask for user features
    train_mask = ~df["is_test"]
    
    # Add user-level features (using only training data)
    df, user_feature_cols = create_user_features(df, train_mask)
    
    # Update to_keep list with user features
    to_keep.extend(user_feature_cols)
    df = df[to_keep]

    # Define hyperparameter search space
    space = {
        'max_depth': hp.uniformint('max_depth', 3, 30),
        'gamma': hp.uniform('gamma', 0, 4),                    # Regularización, suele estar entre 0 y 4
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2), # Típico entre 0.01 y 0.2
        'reg_lambda': hp.uniform('reg_lambda', 0, 4),           # Regularización L2, 0 a 4
        'subsample': hp.uniform('subsample', 0.5, 1),           # Fracción de muestras, 0.5 a 1
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1), # Fracción de columnas, 0.5 a 1
        'n_estimators': hp.uniformint('n_estimators', 10, 150), # Número de árboles, 10 a 150
        'min_child_weight': hp.uniformint('min_child_weight', 1, 10) # Peso mínimo de hijos, 1 a 10
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

    # Usar GroupKFold para evitar data leakage por usuario
    # Necesitamos los grupos de usuario para el GroupKFold
    train_groups = X_train_inicial["username"].values
    
    # Optimización de hiperparametros usando GroupKFold en todo el training data
    best = fmin(
        fn=lambda params: objective_GroupKFold(params, X_train_inicial, pd.Series(y_train_inicial), train_groups),
        space=space,
        algo=tpe.suggest,
        max_evals=MAX_EVALS_BAYESIAN,
        rstate=np.random.default_rng(RAND_SEED_2),
    )
    
    params = space_eval(space, best) # Guardamos los hiperparámetros ganadores.
    print("\nBest hyperparameters found:")
    print(params)

    # Split train vs validation normal (solo para evaluación final)
    # IMPORTANTE: Este split es solo para evaluación final, no para optimización de hiperparámetros
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_inicial, y_train_inicial, test_size=0.2, random_state=RAND_SEED, stratify=y_train_inicial
    )
    
    # Verificar que no hay usuarios compartidos entre train y validation final
    train_users = set(X_train["username"].unique())
    val_users = set(X_val["username"].unique())
    shared_users = train_users & val_users
    if shared_users:
        print(f"WARNING: {len(shared_users)} users appear in both final train and validation sets!")
        print("This could cause overoptimistic validation scores.")
    else:
        print("✓ No users shared between final train and validation sets")

    # Train final model con los mejores hiperparámetros
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
    print(f"  -> Predictions written to 'modelo_benchmark.csv'")

    print("=== Pipeline complete ===")
    end = time.time()
    print(f'Tiempo transcurrido: {str(end - start)} segundos')


if __name__ == "__main__":
    main()
