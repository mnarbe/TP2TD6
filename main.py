import pandas as pd
import time
from database_utils import load_competition_datasets, cast_column_types, momento_del_dia, split_train_test, processFinalInformation
from base_xgboost import trainXGBoostModel
import constants as C
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)

# Adjust this path if needed
PORCENTAJE_DATASET_UTILIZADO = 0.005 # Porcentaje del dataset a utilizar (0.0-1.0)
MAX_EVALS_BAYESIAN = 1 # Cantidad de iteraciones para la optimización bayesiana
FOLD_SPLITS = 3 # Cantidad de folds (KFold o GroupKFold)

def main():
    start = time.time()
    print("=== Starting pipeline ===")

    # Load and preprocess data
    df = load_competition_datasets(
        C.COMPETITION_PATH, sample_frac=PORCENTAJE_DATASET_UTILIZADO, random_state=C.RAND_SEED
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

    # Build feature matrix and get feature names
    test_mask = df["is_test"].to_numpy()
    y = df["target"].to_numpy()
    X = df.drop(columns=["target", "is_test"])

    # Split data
    X_train_dataset, X_test_to_predict, y_train_inicial, _ = split_train_test(X, y, test_mask)

    # Guardar obs_id
    test_obs_ids = X_test_to_predict["obs_id"].copy()
    
    X = X.drop(columns=["obs_id"])

    # Split train vs test
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_dataset, y_train_inicial, test_size=0.2, random_state=C.RAND_SEED, stratify=y_train_inicial
    )

    # Entrenar modelo con XGBoost
    model = trainXGBoostModel(X_train, y_train, FOLD_SPLITS, MAX_EVALS_BAYESIAN)

    # Procesar info final y generar archivo de predicciones
    processFinalInformation(model, X_test, y_test, X_test_to_predict, test_obs_ids)

    print("=== Pipeline complete ===")
    end = time.time()
    print(f'Tiempo transcurrido: {str(end - start)} segundos')

if __name__ == "__main__":
    main()