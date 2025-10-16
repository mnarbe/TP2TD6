import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
import datetime
import numpy as np
import constants as C
import json

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
def processFinalInformation(model, X_test, y_test, X_test_to_predict, test_obs_ids, best_params=None):
    now = datetime.datetime.now()

    # Predict on test set to get validation score
    print("\nGenerating predictions for test set to get final score...")
    preds_val = model.predict_proba(X_test)[:, 1]
    val_score = roc_auc_score(y_test, preds_val)
    print(f"\nValidation ROC AUC: {val_score}")

    # Save params
    if best_params is not None:
        filename_params = f"resultados/modelo_benchmark_{val_score:.3f}_{now.strftime('%Y%m%d_%H%M%S')}_params.json"
        with open(filename_params, 'w') as f:
            json.dump(best_params, f, indent=4)  # indent=4 para que quede legible
        print(f"  --> Params written to '{filename_params}'")

    # Display top feature importances
    print("\nExtracting and sorting feature importances...")
    importances = model.feature_importances_
    imp_series = pd.Series(importances, index=model.get_booster().feature_names)
    imp_series = imp_series.drop(labels=["obs_id"], errors="ignore")
    imp_sorted = imp_series.sort_values(ascending=False)
    filename_imp = f"resultados/modelo_benchmark_{val_score:.3f}_{now.strftime('%Y%m%d_%H%M%S')}_imp.csv"
    imp_sorted.to_csv(filename_imp, header=False)
    print(f"  --> Importances written to '{filename_imp}'")
    print("\nTop feature importances:")
    print(imp_sorted)

    # Generate final test predictions
    print("\nGenerating final predictions for test set...")
    preds_proba = model.predict_proba(X_test_to_predict)[:, 1]
    preds_df = pd.DataFrame({"obs_id": test_obs_ids, "pred_proba": preds_proba})

    # Save final predictions
    filename = f"resultados/modelo_benchmark_{val_score:.3f}_{now.strftime('%Y%m%d_%H%M%S')}.csv"
    preds_df.to_csv(filename, index=False, sep=",")
    print(f"  --> Predictions written to '{filename}'")

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
        "conn_country", "operative_system", "is_mobile", # De contexto
        "is_track", "master_metadata_album_artist_name", "master_metadata_track_name", "track_number", # De canciones
        "is_podcast", "episode_name", "show_name", "show_publisher", "show_total_episodes", # De podcasts
        "track_duration_ms", "podcast_duration_ms", "explicit", "popularity", # Características de la pista
        "time_since_release", "release_date_year", "release_date_month", "song_age_years", # release date,
        "genre1", "genre2", "user_last_song_different", # genre3, # genero
        # "has_popular_artist_genre", "has_rare_artist_genre", "is_kids_genre", "is_comedy_genre", "is_spanish", # genero
        # "has_local_genre", "has_latin_genre", "has_low_energy_genre", "has_high_energy_genre", # genero
        # "has_heavy_genre", "has_party_genre", "has_romantic_genre", "has_relaxing_genre", "has_instrumental_genre", # genero

        # Features de counting
        # Bins por hora
        *[f'hour_bin_{i}' for i in range(24)],
        # Bins por duración
        *[f'duration_bin_{i}' for i in range(10)],
        # Bins por popularidad
        *[f'popularity_bin_{i}' for i in range(5)],
        # Bins por tiempo entre canciones
        *[f'time_diff_bin_{i}' for i in range(6)],
        
        # Contadores derivados
        'artist_play_count',
        # 'genre1_play_count',
        'user_track_listens_today_count',
        'user_artist_listens_today_count',
        # 'daily_songs_bin',
        'user_session_len_so_far',

        # Contadores derivados agregados
        # 'user_session_id',
        # 'time_diff_bin',
        'songs_count_in_day',
        'user_time_since_last_same_track',
        'user_time_since_last_same_artist',
        'user_time_since_last_play',
        'user_track_listens_count',
        
        # Columnas auxiliares necesarias para procesamiento
        "ts", "spotify_track_uri"
    ]

    # Adding time-based features 
    time_based_fields = ["ts", "offline_timestamp"]
    for field in time_based_fields:
        to_keep.extend([
            "month_played_" + field,
            # "hour_of_day_" + field,
            "time_of_day_" + field,
            "year_" + field,
            "weekday_" + field,
            "fin_de_semana_" + field,
            # "day_of_year_" + field,
            # "week_of_year_" + field,
            # "day_of_month_" + field
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
    # Winsorize heavy tails
    tsr_low, tsr_high = df["time_since_release"].quantile([0.01, 0.99])
    df["time_since_release"] = df["time_since_release"].clip(lower=tsr_low, upper=tsr_high)
    df["release_date_year"] = df["release_date"].dt.year.astype("UInt16")
    df["release_date_month"] = df["release_date"].dt.month.astype("UInt8")
    current_year = df['ts'].dt.year
    df['song_age_years'] = current_year - df['release_date_year']
    df["song_age_years"] = df["song_age_years"].astype("UInt8")

    # Add type indicators
    df["is_track"] = df["master_metadata_track_name"].notna().astype("uint8")
    df["is_podcast"] = df["episode_name"].notna().astype("uint8")

    df["track_duration_ms"] = df["duration_ms"].where(df["is_track"] == 1, 0).fillna(0).astype("uint32")
    df["podcast_duration_ms"] = df["duration_ms"].where(df["is_podcast"] == 1, 0).fillna(0).astype("uint64")
    # Winsorize duration
    d_low, d_high = df["duration_ms"].quantile([0.01, 0.995])
    df["track_duration_ms"] = df["track_duration_ms"].clip(lower=d_low, upper=d_high)
    df["podcast_duration_ms"] = df["podcast_duration_ms"].clip(lower=d_low, upper=d_high)

    # Diferencia con la canción anterior escuchada
    df['prev_genre1'] = df.groupby('username', observed=True)['genre1'].shift(1)
    df['user_last_song_different'] = np.where(
        df['genre1'].isna() | df['prev_genre1'].isna(),
        np.nan,
        (df['genre1'] != df['prev_genre1']).astype(int)
    )
    df['user_last_song_different'] = df['user_last_song_different'].fillna(0).astype(np.uint8)
    df.drop(columns='prev_genre1', inplace=True)

    # others
    df['operative_system'] = df['platform'].apply(get_operative_system).astype('category')
    df['is_mobile'] = df['platform'].isin(['ios', 'android']).astype(np.uint8)

    return df

def createNewCountingFeatures(df, df_train):
    # ================
    # Aplica nuevas features de counting a todo el dataset (No genera leakage porque están ordenadas temporalmente)
    # ================

    # Ordeno temporalmente
    df = df.sort_values(['ts'])
    
    # Extraer hora para bin-counting
    df['hour'] = df['ts'].dt.hour
    
    # 1. Bin-counting por hora del día (24 bins) - ACUMULATIVO
    print("  --> Calculating bin-counting for hour of day...")
    for hour in range(24):
        indicator = (df['hour'] == hour).astype(int)
        cum_sum = indicator.groupby(df['username'], observed=True).cumsum() - indicator
        df[f'hour_bin_{hour}'] = cum_sum.astype(np.uint16)
    
    # 2. Bin-counting por duración de canciones (10 bins) - ACUMULATIVO
    print("  --> Calculating bin-counting for track duration...")
    # Crear bins de duración solo en train (sin mirar el futuro)
    duration_quantiles = np.unique(df_train['track_duration_ms'].quantile([i/10 for i in range(11)]).values)
    if len(duration_quantiles) < 2:
        df['duration_bin'] = 0
    else:
        df['duration_bin'] = pd.cut(
            df['track_duration_ms'],
            bins=duration_quantiles,
            labels=False,
            include_lowest=True,
            duplicates='drop'
        )
    if pd.api.types.is_numeric_dtype(df['duration_bin']):
        n_duration_bins = int(pd.Series(df['duration_bin']).dropna().max()) + 1 if pd.Series(df['duration_bin']).notna().any() else 1
        for bin_idx in range(int(n_duration_bins)):
            indicator = (df['duration_bin'] == bin_idx).astype(int)
            cum_sum = indicator.groupby(df['username'], observed=True).cumsum() - indicator
            df[f'duration_bin_{bin_idx}'] = cum_sum.astype(np.uint16)
    
    # 3. Bin-counting por popularidad (5 bins) - ACUMULATIVO
    print("  --> Calculating bin-counting for popularity...")
    popularity_quantiles = np.unique(df_train['popularity'].quantile([i/5 for i in range(6)]).values)
    if len(popularity_quantiles) < 2:
        df['popularity_bin'] = 0
    else:
        df['popularity_bin'] = pd.cut(df['popularity'], bins=popularity_quantiles, labels=False, include_lowest=True, duplicates='drop')
    
    if pd.api.types.is_numeric_dtype(df['popularity_bin']):
        n_pop_bins = int(pd.Series(df['popularity_bin']).dropna().max()) + 1 if pd.Series(df['popularity_bin']).notna().any() else 1
        for bin_idx in range(int(n_pop_bins)):
            indicator = (df['popularity_bin'] == bin_idx).astype(int)
            cum_sum = indicator.groupby(df['username'], observed=True).cumsum() - indicator
            df[f'popularity_bin_{bin_idx}'] = cum_sum.astype(np.uint16)

    # 4. Bin-counting por cantidad de canciones por día (5 bins) - ACUMULATIVO
    print("  --> Calculating bin-counting for daily songs...")
    df['date'] = df['ts'].dt.date
    # Calcular canciones por día HASTA AHORA (histórico)
    df['songs_count_in_day'] = df.groupby(['username', 'date'], observed=True).cumcount()
    # Calcular escuchas del mismo track por usuario en el mismo día HASTA AHORA (histórico)
    df['user_track_listens_today_count'] = df.groupby(['username', 'date', 'spotify_track_uri'], observed=True).cumcount()
    # Calcular escuchas del mismo artista por usuario en el mismo día HASTA AHORA (histórico)
    df['user_artist_listens_today_count'] = df.groupby(['username', 'date', 'master_metadata_album_artist_name'], observed=True).cumcount()

    # 5. Bin-counting por tiempo entre reproducciones (6 bins) - ACUMULATIVO
    print("  --> Calculating bin-counting for time between songs...")
    df['time_between_songs'] = df.groupby('username', observed=True)['ts'].diff().dt.total_seconds()
    time_diff_bins = [-np.inf, 60, 300, 900, 1800, 3600, np.inf]
    df['time_diff_bin'] = pd.cut(df['time_between_songs'], bins=time_diff_bins, labels=False)
    
    for bin_idx in range(6):
        indicator = (df['time_diff_bin'] == bin_idx).astype(int)
        cum_sum = indicator.groupby(df['username'], observed=True).cumsum() - indicator
        df[f'time_diff_bin_{bin_idx}'] = cum_sum.astype(np.uint16)

    # Conteo de reproducciones por artista (acumulado)
    df['artist_play_count'] = df.groupby(['username', 'master_metadata_album_artist_name'], observed=True).cumcount()
    
    # Conteo de reproducciones por género (acumulado)
    df['genre1_play_count'] = df.groupby(['username', 'genre1'], observed=True).cumcount()

    # Recency for same track/artist
    df['user_time_since_last_same_track'] = df.groupby(['username','spotify_track_uri'], observed=True)['ts'].diff().dt.total_seconds().astype(np.float32)
    df['user_time_since_last_same_artist'] = df.groupby(['username','master_metadata_album_artist_name'], observed=True)['ts'].diff().dt.total_seconds().astype(np.float32)

    # Último tiempo de reproducción
    df['user_time_since_last_play'] = df.groupby('username', observed=True)['ts'].diff().dt.total_seconds() / 60
    df['user_time_since_last_play'] = df['user_time_since_last_play'].astype(np.float32)

    # Session boundaries: large gaps or toggles
    session_break = (
        (df['user_time_since_last_play'].fillna(1e9) > 45) |
        (df['offline'].astype('int8').diff().abs().fillna(0) > 0) |
        (df['incognito_mode'].astype('int8').diff().abs().fillna(0) > 0)
    ).astype('int8')
    df['user_session_id'] = session_break.groupby(df['username'], observed=True).cumsum()
    df['user_session_len_so_far'] = df.groupby(['username','user_session_id'], observed=True).cumcount().astype(np.uint16)

    # Counts históricos de escuchas
    df['user_track_listens_count'] = df.groupby(['username', 'spotify_track_uri'], observed=True).cumcount()

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
    
    def processUserProp(df, col_list):
        """
        Calcula la proporción histórica de columnas True/False por usuario,
        hasta antes de la fila actual. Devuelve el DataFrame con todas las columnas añadidas.
        
        col_list: lista de tuplas (col, name) donde col es la columna original y name es el sufijo.
        """
        df = df.copy()
        new_cols = {}

        for col, name in col_list:
            col_val = df[col].astype(float)
            cum_sum = df.groupby('username', observed=True)[col].cumsum() - col_val
            cum_count = df.groupby('username', observed=True)[col].cumcount()
            prop = (cum_sum / cum_count.replace(0, np.nan)).fillna(0.0).astype(np.float32)
            new_cols[f'user_{name}_listened_prop'] = prop

        # Añadir todas las columnas de golpe para evitar fragmentación
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df

    def processUserPropCategorical(df, col, name):
        """
        Calcula la proporción histórica de que cada usuario escuche la categoría
        específica de la fila actual en la columna 'col' hasta justo antes de la observación.
        """
        df = df.copy()
        
        # Convertir a categoría para eficiencia
        df[col] = df[col].astype('category')
        
        categories = df[col].cat.categories
        
        # Crear matriz de proporciones por usuario y categoría
        prop_matrix = pd.DataFrame(index=df.index, columns=categories, dtype=np.float32)
        
        for cat in categories:
            indicator = (df[col] == cat).astype(float)
            cumsum = indicator.groupby(df['username'], observed=True).cumsum() - indicator
            count = df.groupby('username', observed=True).cumcount()
            prop = (cumsum / count.replace(0, np.nan)).fillna(0)
            prop_matrix[cat] = prop
        
        # Seleccionar la proporción correspondiente a la categoría de cada fila
        cat_codes = df[col].cat.codes.to_numpy()
        prop_matrix_np = prop_matrix.to_numpy()
        df[f'user_{name}_listened_prop'] = prop_matrix_np[np.arange(len(df)), cat_codes].astype(np.float32)
        
        return df

    # ================
    # 0 - Features no determinadas por usuario
    # ================
    df = processRate(df, 'operative_system_skip_rate', ['operative_system'])
    df = processRate(df, 'global_month_skip_rate', ['month_played_ts'])
    df = processRate(df, 'track_skip_rate', ['spotify_track_uri'])
    df = processRate(df, 'artist_skip_rate', ['master_metadata_album_artist_name'])

    # ================
    # 1 - Features por usuario
    # ================

    df = processRate(df, 'user_skip_rate', ['username'])

    # Duración promedio de tracks por usuario
    df['user_avg_track_duration_skipped'] = df.groupby('username', observed=True).apply(
        lambda x: (x['track_duration_ms'] * x['target']).expanding().mean().shift()
    ).reset_index(level=0, drop=True).fillna(df['track_duration_ms'].mean()).astype(np.float32)

    # Duración promedio de episodios por usuario
    df['user_avg_podcast_duration_skipped'] = df.groupby('username', observed=True).apply(
        lambda x: (x['podcast_duration_ms'] * x['target']).expanding().mean().shift()
    ).reset_index(level=0, drop=True).fillna(df['podcast_duration_ms'].mean()).astype(np.float32)

    df['user_avg_podcast_duration_not_skipped'] = df.groupby('username', observed=True).apply(
        lambda x: (x['podcast_duration_ms'] * (1 - x['target'])).expanding().mean().shift()
    ).reset_index(level=0, drop=True).fillna(df['podcast_duration_ms'].mean()).astype(np.float32)

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

    df['user_avg_track_duration_not_skipped'] = df.groupby('username', observed=True).apply(
        lambda x: (x['track_duration_ms'] * (1 - x['target'])).expanding().mean().shift()
    ).reset_index(level=0, drop=True).fillna(df['track_duration_ms'].mean()).astype(np.float32)

    # ================
    # 2 - Por usuario + operative_system
    # ================
    df = processRate(df, 'user_operative_system_skip_rate', ['username', 'operative_system'])

    # ================
    # 3 - Por usuario + flags booleanas
    # ================
    df = processRate(df, 'user_explicit_skip_rate', ['username', 'explicit'])
    df = processRate(df, 'user_hour_skip_rate', ['username', 'hour_of_day_ts'])
    df = processRate(df, 'user_weekday_skip_rate', ['username', 'weekday_ts'])
    df = processRate(df, 'user_known_artist_skip_rate', ['username', 'master_metadata_album_artist_name'])
    df = processRate(df, 'user_shuffled_skip_rate', ['username', 'shuffle'])
    df = processRate(df, 'user_offline_skip_rate', ['username', 'offline'])
    
    # Defragment after many column insertions to improve performance
    df = df.copy()

    return df

def applyHistoricalFeaturesToSet(df_target, df_train):
    df_target = applyHistoricalNonUserFeaturesToSet(df_target, df_train)
    df_target = applyHistoricalUserFeaturesToSet(df_target, df_train)
    return df_target

def applyHistoricalUserFeaturesToSet(df_target, df_train):
    """
    Toma las features históricas generadas en el TRAIN y aplica
    el último valor observado (temporalmente consistente)
    a cada fila de df_target, respetando las features condicionales
    (por username y otra variable).
    """
    df_target = df_target.copy()

    # Asegurar orden temporal
    df_train = df_train.sort_values(['username', 'ts'], ascending=True)

    # Helper que devuelve el último valor por grupo
    def last_by_group(group_cols, feature_cols):
        return (
            df_train.groupby(group_cols, observed=True)
            .tail(1)
            .set_index(group_cols)[feature_cols]
        )

    # ================
    # 1️⃣ Features por usuario
    # ================
    user_cols = [
        'user_skip_rate',
        'user_avg_track_duration_skipped',
        'user_avg_track_duration_not_skipped',
        'user_avg_podcast_duration_skipped',
        'user_avg_podcast_duration_not_skipped',
        'user_preference_this_age',
        'user_time_since_last_play',
        'user_time_since_last_same_track',
        'user_time_since_last_same_artist',
        'user_track_listens_count',
        'artist_play_count',
        'genre1_play_count',
        'user_artist_listens_today_count',
        'user_track_listens_today_count',
        'user_session_len_so_far',
        
    ]
    df_target = df_target.merge(
        last_by_group(['username'], user_cols),
        on=['username'], how='left'
    )

    # ================
    # 2️⃣ Por usuario + operative_system
    # ================
    os_cols = ['user_operative_system_skip_rate']
    df_target = df_target.merge(
        last_by_group(['username', 'operative_system'], os_cols),
        on=['username', 'operative_system'], how='left'
    )

    # ================
    # 3️⃣ Por usuario + flags booleanas
    # ================
    boolean_groups = {
        'user_explicit_skip_rate': ['username', 'explicit'],
        'user_hour_skip_rate': ['username', 'hour_of_day_ts'],
        'user_weekday_skip_rate': ['username', 'weekday_ts'],
        'user_known_artist_skip_rate': ['username', 'master_metadata_album_artist_name'],
        'user_shuffled_skip_rate': ['username', 'shuffle'],
        'user_offline_skip_rate': ['username', 'offline'],
    }

    for feat, group_cols in boolean_groups.items():
        df_target = df_target.merge(
            last_by_group(group_cols, [feat]),
            on=group_cols, how='left'
        )

    
    return df_target

def applyHistoricalNonUserFeaturesToSet(df_target, df_train):
    """
    Aplica features históricas del set de train a df_target para:
    - operative_system_skip_rate
    - global_month_skip_rate
    - track_skip_rate
    - artist_skip_rate

    df_target: dataframe donde queremos aplicar los features
    df_train: dataframe de entrenamiento ya procesado con createNewSetFeatures
    """
    df_target = df_target.copy()
    global_mean = df_train['target'].mean()

    def apply_historical_rate(df_target, df_train, group_col, feature_name, global_mean):
        tmp = df_train.groupby(group_col, observed=True)['target'].cumsum() - df_train['target']
        count = df_train.groupby(group_col, observed=True).cumcount()
        rate = tmp / count.replace(0, np.nan)
        rate_map = rate.groupby(df_train[group_col], observed=True).last().fillna(global_mean)
        df_target[feature_name] = df_target[group_col].map(rate_map).astype(np.float32)
        return df_target

    df_target = apply_historical_rate(df_target, df_train, 'operative_system', 'operative_system_skip_rate', global_mean)
    if 'month_played_ts' in df_target.columns:
        df_target = apply_historical_rate(df_target, df_train, 'month_played_ts', 'global_month_skip_rate', global_mean)
    df_target = apply_historical_rate(df_target, df_train, 'spotify_track_uri', 'track_skip_rate', global_mean)
    df_target = apply_historical_rate(df_target, df_train, 'master_metadata_album_artist_name', 'artist_skip_rate', global_mean)

    return df_target

def createNewTimeBasedFeatures(df, field):
    df["month_played_" + field] = df[field].dt.month.astype("uint8")
    df["hour_of_day_" + field] = df[field].dt.hour.astype("uint8")
    df["time_of_day_" + field] = df[field].dt.hour.apply(momento_del_dia).astype("category")
    df["year_" + field] = df[field].dt.year.astype("uint16")
    df["weekday_" + field] = df[field].dt.weekday.astype("uint8")
    df["fin_de_semana_" + field] = df[field].apply(es_finde)
    df["day_of_year_" + field] = df[field].dt.dayofyear.astype("uint16")
    df["day_of_month_" + field] = df[field].dt.day.astype("uint8")

    return df

def createNewTimeBasedFeaturesSimple(df, field):
    df["month_played_" + field] = df[field].dt.month.astype("uint8")
    df["hour_of_day_" + field] = df[field].dt.hour.astype("uint8")
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

import numpy as np

def get_operative_system(p):
    if pd.isna(p):
        return np.nan
    p = p.lower()

    # ORDEN IMPORTANTE → más específicos primero
    if 'apple' in p and 'tv' in p:
        return 'apple_tv'
    elif 'cast' in p:
        return 'cast'
    elif 'ios' in p or 'apple' in p:
        return 'ios'
    elif 'android' in p:
        return 'android'
    elif 'tizen' in p or ('tv' in p):
        return 'tv'
    elif 'osx' in p or 'os x' in p:
        return 'osx'
    elif 'windows' in p:
        return 'windows'
    elif 'linux' in p:
        return 'linux'
    else:
        return 'other'


def simple_clustering(df, kmeans_model=None, n_clusters=3):
    """Versión corregida que permite usar un modelo existente o entrenar uno nuevo"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['obs_id', 'year_ts', 'target']]
    
    X_numeric = df[numeric_cols].fillna(0)
    
    if kmeans_model is None:
        # Entrenar nuevo modelo (solo para train)
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=C.RAND_SEED, n_init=20)
        df['cluster'] = kmeans_model.fit_predict(X_numeric).astype('int')
    else:
        # Usar modelo existente (para validation/test)
        df['cluster'] = kmeans_model.predict(X_numeric).astype('int')
    
    return df, kmeans_model