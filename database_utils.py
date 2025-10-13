import os
import pandas as pd
from sklearn.metrics import roc_auc_score
import datetime
import numpy as np

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

# Split train test
def split_train_test(df):
    """
    Split features and labels into train/test based on mask.
    """
    print("Splitting data into train/test sets...")

    # First, separate the actual test set (unknown labels)
    test_mask = df["is_test"].to_numpy()
    X, y = split_x_and_y(df)

    train_mask = ~test_mask  # Invertir la máscara

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    print(f"  --> Training set: {X_train.shape[0]} rows")
    print(f"  --> Test set:     {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test

def split_train_test_df(df):
    """
    Split features and labels into train/test based on mask.
    """
    print("Splitting data into train/test sets...")

    # First, separate the actual test set (unknown labels)
    test_mask = df["is_test"].to_numpy()
    train_mask = ~test_mask  # Invertir la máscara

    df_train = df[train_mask]
    df_test = df[test_mask]

    print(f"  --> Training set: {df_train.shape[0]} rows")
    print(f"  --> Test set:     {df_test.shape[0]} rows")
    return df_train, df_test

# Split x and y
def split_x_and_y(df):
    y = df["target"].to_numpy()
    X = df.drop(columns=["target", "is_test"])
    return X, y

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
        "popularity": "Int16",
        "show_name": "category",
        "show_publisher": "category",
        "track_number": "Int16",
        "show_total_episodes": "Int16",
        # Nuevas columnas de genero
        "genre1": "category", "genre2": "category", "genre3": "category",
        "has_popular_artist_genre": bool, "has_rare_artist_genre": bool,
        "is_kids_genre": bool,
        "is_comedy_genre": bool,
        "is_spanish": bool,
        "has_japanese_genres": bool,
        "has_local_genre": bool,
        "has_latin_genre": bool,
        "has_low_energy_genre": bool,
        "has_high_energy_genre": bool,
        "has_heavy_genre": bool,
        "has_party_genre": bool,
        "has_romantic_genre": bool,
        "has_relaxing_genre": bool,
        "has_instrumental_genre": bool
    }

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["offline_timestamp"] = pd.to_datetime(df["ts"], utc=True)
    df = df.astype(dtype_map)
    print("  --> Column types cast successfully.")
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
        "time_since_release", "release_date_year", "release_date_month", "song_age_years", # release date,
        "genre1", "genre2", "genre3", # genero
        "has_popular_artist_genre", "has_rare_artist_genre", "is_kids_genre", "is_comedy_genre", "is_spanish", # genero
        "has_japanese_genres", "has_local_genre", "has_latin_genre", "has_low_energy_genre", "has_high_energy_genre", # genero
        "has_heavy_genre", "has_party_genre", "has_romantic_genre", "has_relaxing_genre", "has_instrumental_genre", # genero
        "ts", "spotify_track_uri" # Conservar auxiliares para procesamiento futuro
    ]

    # Adding time-based features 
    time_based_fields = ["ts", "offline_timestamp"]
    for field in time_based_fields:
        to_keep.extend([
            "month_played_" + field,
            "hour_of_day_" + field,
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

# FEATURE ENGINEERING

def createNewFeatures(df):
    # Add time-based features on timestamp
    df = createNewTimeBasedFeatures(df, "ts")
    df = createNewTimeBasedFeaturesSimple(df, "offline_timestamp")

    # Add release date features
    df["release_date"] = df["release_date"].combine_first(df["album_release_date"])
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce", utc=True)
    df["time_since_release"] = (df["ts"] - df["release_date"]).dt.total_seconds()
    df["time_since_release"] = df["time_since_release"].astype("float32")
    df["release_date_year"] = df["release_date"].dt.year.astype("UInt16")
    df["release_date_month"] = df["release_date"].dt.month.astype("UInt8")
    current_year = df['ts'].dt.year
    df['song_age_years'] = current_year - df['release_date_year']
    df["song_age_years"] = df["song_age_years"].astype("UInt8")

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

def createNewSetFeatures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features históricas (acumuladas hasta antes de la observación actual)
    para el dataframe de TRAIN. Asume:
      - df ya está ordenado por ['username', 'ts'] (ascendente).
      - df contiene la columna 'target' (0/1 skip).
      - ts es datetime (si no, intenta convertirlo).
    Añade columnas:
      - user_skip_rate
      - user_operative_system_skip_rate
      - track_skip_rate
      - artist_skip_rate
      - user_explicit_skip_rate
      - user_hour_skip_rate
    """
    df = df.copy()

    # Asegurar ts datetime
    if not pd.api.types.is_datetime64_any_dtype(df['ts']):
        df['ts'] = pd.to_datetime(df['ts'])
    else:
        # si tiene tz, convertir a naive
        if df['ts'].dt.tz is not None:
            df['ts'] = df['ts'].dt.tz_convert(None)

    # base global mean para imputar donde no hay pasado
    global_mean = df['target'].mean()

    # ---------------------------
    # Helper para calcular rates
    # ---------------------------
    def historical_rate(df, group_cols, target_col='target', out_name='rate'):
        """
        Calcula (cum_skips_before / cum_count_before) por grupo definido en group_cols,
        vectorizado sin usar .shift() para evitar reindexados lentos.
        Retorna un Series con el rate.
        """
        # suma acumulada INCLUYENDO la fila actual
        cum_skips_incl = df.groupby(group_cols, observed=True)[target_col].cumsum()
        # cuenta acumulada INCLUYENDO la fila actual (1,2,3,...)
        cum_count_incl = df.groupby(group_cols, observed=True).cumcount() + 1

        # pasar a "hasta antes de la fila actual"
        cum_skips_before = cum_skips_incl - df[target_col]
        cum_count_before = cum_count_incl - 1

        # evitar división por 0: donde count_before == 0 -> NaN
        rate = cum_skips_before / cum_count_before
        rate = rate.where(cum_count_before > 0, np.nan)

        # rellenar NaN con global_mean (puedes cambiar la estrategia)
        rate = rate.fillna(global_mean)

        rate.name = out_name
        return rate
    
    def processRate(df, col, variables):
        df[col] = historical_rate(df, variables, out_name=col)
        df[col] = df[col].astype(np.float32)  # fuerza tipo numérico

        return df

    # Skip rates básicos
    df = processRate(df, 'global_month_skip_rate', ['month_played_ts'])
    df = processRate(df, 'track_skip_rate', ['spotify_track_uri'])
    df = processRate(df, 'artist_skip_rate', ['master_metadata_album_artist_name'])
    df = processRate(df, 'user_skip_rate', ['username'])
    df = processRate(df, 'user_operative_system_skip_rate', ['username', 'operative_system'])
    df = processRate(df, 'user_explicit_skip_rate', ['username', 'explicit'])
    df = processRate(df, 'user_hour_skip_rate', ['username', 'hour_of_day_ts'])
    df = processRate(df, 'user_weekday_skip_rate', ['username', 'weekday_ts'])
    df = processRate(df, 'user_known_artist_skip_rate', ['username', 'master_metadata_album_artist_name'])
    df = processRate(df, 'user_shuffled_skip_rate', ['username', 'shuffle'])
    df = processRate(df, 'user_offline_skip_rate', ['username', 'offline'])

    # Duración promedio de tracks por usuario
    df['user_avg_track_duration_skipped'] = df.groupby('username').apply(
        lambda x: (x['duration_ms'] * x['target']).expanding().mean().shift()
    ).reset_index(level=0, drop=True).fillna(df['duration_ms'].mean()).astype(np.float32)

    df['user_avg_track_duration_not_skipped'] = df.groupby('username').apply(
        lambda x: (x['duration_ms'] * (1 - x['target'])).expanding().mean().shift()
    ).reset_index(level=0, drop=True).fillna(df['duration_ms'].mean()).astype(np.float32)

    # Preferencias según antigüedad de canción
    age_buckets = {
        'new': (0, 3),
        'old': (3, 20),
        'extremely_old': (20, 100)
    }
    df['user_preference_this_age'] = np.nan
    for low, high in age_buckets.values():
        mask = df['song_age_years'].between(low, high, inclusive='left')
        
        if mask.any():
            # Historial de skips del usuario solo para ese bucket
            tmp = df.loc[mask].groupby('username', observed=True)['target'].cumsum() - df.loc[mask, 'target']
            count = df.loc[mask].groupby('username', observed=True).cumcount()
            rate = tmp / count.replace(0, np.nan)
            
            # Asignamos el rate histórico
            df.loc[mask, 'user_preference_this_age'] = rate.fillna(global_mean).astype(np.float32)
    df['user_preference_this_age'] = df['user_preference_this_age'].fillna(global_mean)

    # Último tiempo de reproducción
    df['user_time_since_last_play'] = df.groupby('username')['ts'].diff().dt.total_seconds() / 60
    df['user_time_since_last_play'] = df['user_time_since_last_play'].astype(np.float32)

    # Consumo fin de semana por artista
    g = df.groupby(['username', 'master_metadata_album_artist_name'], observed=True)['weekday_ts']
    df['user_artist_weekend_consumed'] = (g.cummax() >= 5).astype(np.uint8)

    # Track / album / artist / genre vistos antes
    df['user_track_seen_before'] = df.groupby('username')['spotify_track_uri'].cumcount().gt(0).astype(np.uint8)
    df['user_artist_seen_before'] = df.groupby('username')['master_metadata_album_artist_name'].cumcount().gt(0).astype(np.uint8)

    # Counts históricos de escuchas
    df['user_artist_listens_count'] = df.groupby(['username', 'master_metadata_album_artist_name']).cumcount()
    df['user_track_listens_count'] = df.groupby(['username', 'spotify_track_uri']).cumcount()

    return df

def applyHistoricalFeaturesToSet(df_target, df_train):
    df_target = applyHistoricalNonUserFeaturesToSet(df_target, df_train)
    df_target = applyHistoricalUserFeaturesToSet(df_target, df_train)
    return df_target

def applyHistoricalUserFeaturesToSet(df_target, df_train):
    df_target = df_target.copy()
    
    # Base de datos de últimos valores por usuario
    user_last = (
        df_train.sort_values(['username', 'ts'])
        .groupby('username')
        .tail(1)
        .set_index('username')
        [[
            'user_skip_rate',
            'user_operative_system_skip_rate',
            'user_explicit_skip_rate',
            'user_hour_skip_rate',
            'user_weekday_skip_rate',
            'user_known_artist_skip_rate',
            'user_shuffled_skip_rate',
            'user_offline_skip_rate',
            'user_avg_track_duration_skipped',
            'user_avg_track_duration_not_skipped',
            'user_preference_this_age',
            'user_time_since_last_play',
            'user_artist_weekend_consumed',
            'user_track_seen_before',
            'user_artist_seen_before',
            'user_artist_listens_count',
            'user_track_listens_count'
        ]]
    )

    # Merge
    df_target = df_target.merge(user_last, on='username', how='left')

    return df_target

def applyHistoricalNonUserFeaturesToSet(df_target, df_train):
    """
    Aplica features históricas del set de train a df_target para:
    - global_month_skip_rate
    - track_skip_rate
    - artist_skip_rate

    df_target: dataframe donde queremos aplicar los features
    df_train: dataframe de entrenamiento ya procesado con createNewSetFeatures
    """
    df_target = df_target.copy()
    global_mean = df_train['target'].mean()

    # 1) global_month_skip_rate
    if 'month_played_ts' in df_target.columns:
        tmp = df_train.groupby('month_played_ts', observed=True)['target'].cumsum() - df_train['target']
        count = df_train.groupby('month_played_ts', observed=True).cumcount()
        rate = tmp / count.replace(0, np.nan)
        month_rate_map = rate.groupby(df_train['month_played_ts']).last().fillna(global_mean)
        df_target['global_month_skip_rate'] = df_target['month_played_ts'].map(month_rate_map).astype(np.float32)

    # 2) track_skip_rate
    tmp = df_train.groupby('spotify_track_uri', observed=True)['target'].cumsum() - df_train['target']
    count = df_train.groupby('spotify_track_uri', observed=True).cumcount()
    rate = tmp / count.replace(0, np.nan)
    track_rate_map = rate.groupby(df_train['spotify_track_uri']).last().fillna(global_mean)
    df_target['track_skip_rate'] = df_target['spotify_track_uri'].map(track_rate_map).astype(np.float32)

    # 3) artist_skip_rate
    tmp = df_train.groupby('master_metadata_album_artist_name', observed=True)['target'].cumsum() - df_train['target']
    count = df_train.groupby('master_metadata_album_artist_name', observed=True).cumcount()
    rate = tmp / count.replace(0, np.nan)
    artist_rate_map = rate.groupby(df_train['master_metadata_album_artist_name']).last().fillna(global_mean)
    df_target['artist_skip_rate'] = df_target['master_metadata_album_artist_name'].map(artist_rate_map).astype(np.float32)

    return df_target

def createNewTimeBasedFeatures(df, field):
    df["month_played_" + field] = df[field].dt.month.astype("uint8")
    df["hour_of_day_" + field] = df[field].dt.month.astype("uint8")
    df["time_of_day_" + field] = df[field].dt.hour.apply(momento_del_dia).astype("category")
    df["year_" + field] = df[field].dt.year.astype("uint16")
    df["weekday_" + field] = df[field].dt.weekday.astype("uint8")
    df["fin_de_semana_" + field] = df[field].apply(es_finde)
    df["day_of_year_" + field] = df[field].dt.dayofyear.astype("uint16")
    df["day_of_month_" + field] = df[field].dt.day.astype("uint8")

    return df

def createNewTimeBasedFeaturesSimple(df, field):
    df["month_played_" + field] = df[field].dt.month.astype("uint8")
    df["hour_of_day_" + field] = df[field].dt.month.astype("uint8")
    df["time_of_day_" + field] = df[field].dt.hour.apply(momento_del_dia).astype("category")
    df["year_" + field] = df[field].dt.year.astype("uint16")
    df["day_of_year_" + field] = df[field].dt.dayofyear.astype("uint16")
    df["day_of_month_" + field] = df[field].dt.day.astype("uint8")

    return df


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