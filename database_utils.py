import os
import pandas as pd
from sklearn.metrics import roc_auc_score
import datetime

# Carga del dataset
def load_competition_datasets(data_dir, sample_frac=None, random_state=None):
    """
    Load train and test datasets, optionally sample a fraction of the training set,
    concatenate, and reset index.
    """
    print("Loading competition datasets from:", data_dir)
    train_file = os.path.join(data_dir, "merged_data.csv")
    test_file = os.path.join(data_dir, "merged_test_data.csv")

    # Load training data and optionally subsample
    train_df = pd.read_csv(train_file, low_memory=False)
    if sample_frac is not None:
        train_df = train_df.sample(frac=sample_frac, random_state=random_state)

    # Load test data
    test_df = pd.read_csv(test_file, low_memory=False)

    # Concatenate and reset index
    combined = pd.concat([train_df, test_df], ignore_index=True)
    print(f"  --> Concatenated DataFrame: {combined.shape[0]} rows")
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

def es_finde(dia):
    return dia.weekday() >= 5

def cast_column_types(df):
    """
    Cast columns to efficient dtypes and parse datetime fields.
    """
    print("Casting column types and parsing datetime fields...")
    dtype_map = {
        "conn_country": "category",
        "ip_addr": "category",
        "master_metadata_track_name": "category",
        "master_metadata_album_artist_name": "category",
        "master_metadata_album_album_name": "category",
        "reason_end": "category",
        "username": "category",
        "episode_name": "category",
        "episode_show_name": "category",
        "spotify_episode_uri": "string",
        "shuffle": bool,
        "offline": bool,
        "incognito_mode": bool,
        "obs_id": int,
        "fin_de_semana": bool,
        # Nuevas columnas de mergecsv - convertir a tipos apropiados
        "explicit": bool,
        "release_date": "category",
        "album_release_date": "category",
        "popularity": "Int32",
        "show_name": "category",
        "show_publisher": "category",
        "track_number": "Int32",
        "is_short_track": "bool",
        "is_long_track": "bool",
        "show_total_episodes": "Int32"
    }

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["offline_timestamp"] = pd.to_datetime(
        df["offline_timestamp"], unit="s", errors="coerce", utc=True
    )
    df = df.astype(dtype_map)
    print("  --> Column types cast successfully.")
    return df

def split_train_test(X, y, test_mask):
    """
    Split features and labels into train/test based on mask.
    """
    print("Splitting data into train/test sets...")

    # First, separate the actual test set (unknown labels)
    test_mask = df["is_test"].to_numpy()
    y = df["target"].to_numpy()
    X = df.drop(columns=["target", "is_test"])

    train_mask = ~test_mask  # Invertir la máscara

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    print(f"  --> Training set: {X_train.shape[0]} rows")
    print(f"  --> Test set:     {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test

# Procesado luego de entrenar el modelo y obtener los mejores hiperparámetros
def processFinalInformation(model, X_test, y_test, X_test_to_predict, test_obs_ids):
    # Display top feature importances
    print("\nExtracting and sorting feature importances...")
    importances = model.feature_importances_
    imp_series = pd.Series(importances, index=model.get_booster().feature_names)
    imp_series = imp_series.drop(labels=["obs_id"], errors="ignore")
    imp_sorted = imp_series.sort_values(ascending=False)
    print("\nTop feature importances:")
    print(imp_sorted)

    # Predict on test set to get validation score
    print("\nGenerating predictions for test set to get final score...")
    preds_val = model.predict_proba(X_test)[:, 1]
    val_score = roc_auc_score(y_test, preds_val)
    print(f"\nValidation ROC AUC: {val_score}")

    # Generate final test predictions
    print("\nGenerating final predictions for test set...")
    preds_proba = model.predict_proba(X_test_to_predict)[:, 1]
    preds_df = pd.DataFrame({"obs_id": test_obs_ids, "pred_proba": preds_proba})
    preds_df.to_csv("modelo_benchmark.csv", index=False, sep=",")
    print(f"  --> Predictions written to 'modelo_benchmark.csv'")

#######################
# PRE PROCESAMIENTO
#######################

def cast_column_types(df):
    """
    Cast columns to efficient dtypes and parse datetime fields.
    """
    print("Casting column types and parsing datetime fields...")
    dtype_map = {
        "conn_country": "category",
        "ip_addr": "category",
        "master_metadata_track_name": "category",
        "master_metadata_album_artist_name": "category",
        "master_metadata_album_album_name": "category",
        "reason_end": "category",
        "username": "category",
        "episode_name": "category",
        "episode_show_name": "category",
        "spotify_episode_uri": "string",
        "shuffle": bool,
        "offline": bool,
        "incognito_mode": bool,
        "obs_id": int,
        # Nuevas columnas de mergecsv - convertir a tipos apropiados
        "explicit": bool,
        "popularity": "Int32",
        "show_name": "category",
        "show_publisher": "category",
        "track_number": "Int32",
        "show_total_episodes": "Int32"
    }

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["offline_timestamp"] = pd.to_datetime(df["ts"], utc=True)
    df = df.astype(dtype_map)
    print("  --> Column types cast successfully.")
    return df

def createNewFeatures(df):
    # Add time-based features on timestamp
    df = createNewTimeBasedFeatures(df, "ts")
    df = createNewTimeBasedFeaturesSimple(df, "offline_timestamp")

    # Add release date features
    df["release_date"] = df["release_date"].combine_first(df["album_release_date"])
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce", utc=True)
    df["time_since_release"] = (df["ts"] - df["release_date"]).dt.total_seconds()
    df["release_date_year"] = df["release_date"].dt.year.astype("UInt16")
    df["release_date_month"] = df["release_date"].dt.month.astype("UInt16")

    # Add type indicators
    df["is_track"] = df["master_metadata_track_name"].notna().astype("uint8")
    df["is_podcast"] = df["episode_name"].notna().astype("uint8")

    # Add track duration indicators
    df["is_short_track"] = (
        (df["duration_ms"].notna()) &
        (df["duration_ms"] <= 90000) &
        (df["is_track"] == True)
    ).astype("bool")
    df["is_long_track"] = (
        (df["duration_ms"].notna()) &
        (df["duration_ms"] > 360000) &
        (df["is_track"] == True)
    ).astype("bool")

    # others
    df["operative_system"] = df["platform"].str.strip().str.split(n=1).str[0].astype("category")

    return df

def createNewTimeBasedFeatures(df, field):
    df["month_played_" + field] = df[field].dt.month.astype("uint8")
    df["time_of_day_" + field] = df[field].dt.hour.apply(momento_del_dia).astype("category")
    df["year_" + field] = df[field].dt.year.astype("uint16")
    df["weekday_" + field] = df[field].dt.weekday.astype("uint8")
    df["fin_de_semana_" + field] = df[field].apply(es_finde)
    df["day_of_year_" + field] = df[field].dt.dayofyear.astype("uint16")
    df["day_of_month_" + field] = df[field].dt.day.astype("uint8")

    return df

def createNewTimeBasedFeaturesSimple(df, field):
    df["month_played_" + field] = df[field].dt.month.astype("uint8")
    df["time_of_day_" + field] = df[field].dt.hour.apply(momento_del_dia).astype("category")
    df["year_" + field] = df[field].dt.year.astype("uint16")
    df["day_of_year_" + field] = df[field].dt.dayofyear.astype("uint16")
    df["day_of_month_" + field] = df[field].dt.day.astype("uint8")

    return df

def processTargetAndTestMask(df):
    # Create target and test mask
    print("Creating 'target' and 'is_test' columns...")
    df["target"] = (df["reason_end"] == "fwdbtn").astype(int)
    df["is_test"] = df["reason_end"].isna()
    df.drop(columns=["reason_end"], inplace=True)
    print("  --> 'target' and 'is_test' created, dropped 'reason_end' column.")

    return df

def keepImportantColumnsDefault(df):
    # Keep only relevant columns (including year for temporal split)
    to_keep = [
        "obs_id", "username", "ip_addr",
        "target", "is_test",
        "incognito_mode", "offline", "shuffle", # Booleanas
        "conn_country", "operative_system", # De contexto
        "is_track", "master_metadata_album_artist_name", "master_metadata_track_name", "track_number", # De canciones
        "is_podcast", "episode_name", "show_name", "show_publisher", "show_total_episodes", # De podcasts
        "is_short_track", "is_long_track", "duration_ms", "explicit", "popularity", # Características de la pista
        "time_since_release", "release_date_year", "release_date_month" # release date
    ]

    # Adding time-based features 
    time_based_fields = ["ts", "offline_timestamp"]
    for field in time_based_fields:
        to_keep.extend([
            "month_played_" + field,
            "time_of_day_" + field,
            "year_" + field,
            "weekday_" + field,
            "fin_de_semana_" + field,
            "day_of_year_" + field,
            "week_of_year_" + field,
            "day_of_month_" + field
        ])

    # Keep only existing columns
    return df[[col for col in to_keep if col in df.columns]]

#######################
# MÉTODOS AUXILIARES
#######################

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

def es_finde(dia):
    return dia.weekday() >= 5
