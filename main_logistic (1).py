#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_logistic.py
----------------
Entrena un modelo de Regresión Logística para el mismo objetivo binario
que tu pipeline de XGBoost y genera un archivo de predicciones 'modelo_logistica.csv'
para comparar rendimientos (ROC AUC y ranking de coeficientes).

Incluye manejo de valores faltantes (NaN):
- Categóricas: el missing se considera como nueva categoría ("__MISSING__")
- Numéricas: imputación por media
- Booleanas: imputación probabilística (1 con prob. igual a la proporción de 1s por columna)
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

# Utils y constantes de tu repo
from database_utils import (
    load_competition_datasets,
    cast_column_types,
    momento_del_dia,
    split_train_test,
)
import constants as C

pd.set_option("display.max_columns", None)

# Hiperparámetros principales
PORCENTAJE_DATASET_UTILIZADO = 0.005  # 0.0-1.0
TEST_SIZE = 0.20
RANDOM_STATE = C.RAND_SEED
LOGREG_MAX_ITER = 500

class ProbabilisticBooleanImputer(BaseEstimator, TransformerMixin):
    """
    Imputa NaN en columnas booleanas con 1 con probabilidad p = proporción de 1s observados (por columna).
    Soporta entrada como DataFrame o ndarray. Devuelve ndarray (para ColumnTransformer).
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.ps_ = None
        self.columns_: List[str] = None
        self._rng = np.random.RandomState(self.random_state)

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns.tolist()
            arr = X.astype('float').to_numpy()  # bool/NaN -> float
        else:
            arr = X.astype('float')
            self.columns_ = [f"col_{i}" for i in range(arr.shape[1])]
        # p = mean de True ignorando NaN
        with np.errstate(invalid="ignore"):
            p = np.nanmean(arr, axis=0)
        # Si una columna es todo NaN, usar p=0.5 como neutro
        p = np.where(np.isnan(p), 0.5, p)
        # Limitar al rango [0,1]
        p = np.clip(p, 0.0, 1.0)
        self.ps_ = p
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            arr = X.astype('float').to_numpy()
        else:
            arr = X.astype('float')
        mask_nan = np.isnan(arr)
        if mask_nan.any():
            # sample por columna
            n_rows, n_cols = arr.shape
            # generar muestras Bernoulli para todas las celdas NaN
            samples = self._rng.binomial(1, self.ps_, size=(n_rows, n_cols)).astype(float)
            arr[mask_nan] = samples[mask_nan]
        # Devolver como int/bool 0/1
        return arr.astype(int)

def main():
    t0 = time.time()
    print("=== Starting LOGISTIC REGRESSION pipeline ===")

    # 1) Carga y preprocesamiento base (idéntico a main.py)
    df = load_competition_datasets(
        C.COMPETITION_PATH, sample_frac=PORCENTAJE_DATASET_UTILIZADO, random_state=C.RAND_SEED
    )
    df = cast_column_types(df)

    # Feature engineering idéntico
    df["month_played"] = df["ts"].dt.month.astype("uint8")
    df["time_of_day"] = df["ts"].dt.hour.apply(momento_del_dia).astype("category")
    df["is_track"] = df["master_metadata_track_name"].notna().astype("uint8")
    df["is_podcast"] = df["episode_name"].notna().astype("uint8")
    df["operative_system"] = df["platform"].str.strip().str.split(n=1).str[0].astype("category")
    df = df.sort_values(["obs_id"])

    # 2) Target e is_test (idéntico a main.py)
    print("Creating 'target' and 'is_test' columns...")
    df["target"] = (df["reason_end"] == "fwdbtn").astype(int)
    df["is_test"] = df["reason_end"].isna()
    df.drop(columns=["reason_end"], inplace=True)
    print("  → 'target' and 'is_test' created, dropped 'reason_end'.")

    # 3) Subset de columnas como en main.py (robusto a ausentes)
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
        "time_of_day",
        "is_track",
        "is_podcast",
        "operative_system",
        # De mergecsv
        "name", "duration_ms", "explicit", "release_date",
        "album_name", "album_release_date", "artist_name", "popularity",
        "track_number", "show_name", "show_publisher", "show_total_episodes",
    ]
    to_keep = [c for c in to_keep if c in df.columns]
    df = df[to_keep]

    # Tipos
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category")

    bool_cols = ["explicit", "incognito_mode", "offline", "shuffle", "is_track", "is_podcast"]
    for col in bool_cols:
        if col in df.columns:
            # Mantener como booleano con NaN posibles (pandas usa dtype object/bool con NaN)
            df[col] = df[col].astype('float')  # pasamos a float para permitir NaN; luego el imputador devuelve 0/1
            # no rellenamos aquí; lo hará el imputador probabilístico
            # (si ya vienen sin NaN, no afecta)

    # 4) Construcción de matrices y split "competición"
    test_mask = df["is_test"].to_numpy()
    y = df["target"].to_numpy()
    X = df.drop(columns=["target", "is_test"])

    X_train_dataset, X_test_to_predict, y_train_inicial, _ = split_train_test(X, y, test_mask)

    test_obs_ids = X_test_to_predict["obs_id"].copy()

    X = X.drop(columns=["obs_id"])
    X_train_dataset = X_train_dataset.drop(columns=["obs_id"])
    X_test_to_predict_features = X_test_to_predict.drop(columns=["obs_id"])

    # 5) Train/Val split interno
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_dataset,
        y_train_inicial,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_inicial,
    )

    # 6) Columnas por tipo
    categorical_cols = X_train.select_dtypes(include=["category"]).columns.tolist()
    all_cols = X_train.columns.tolist()
    # booleanas: las que definimos en bool_cols y existan
    bool_in_cols = [c for c in bool_cols if c in X_train.columns]
    numeric_cols = [c for c in all_cols if c not in categorical_cols and c not in bool_in_cols]

    # Compatibilidad con distintas versiones de scikit-learn
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    # 7) Preprocesamiento con imputación
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("ohe", ohe),
    ])

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    bool_pipeline = Pipeline(steps=[
        ("imputer_bool", ProbabilisticBooleanImputer(random_state=RANDOM_STATE)),
        # nada más; ya devuelve 0/1
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, categorical_cols),
            ("num", num_pipeline, numeric_cols),
            ("bool", bool_pipeline, bool_in_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    # 8) Modelo de Regresión Logística
    logreg = LogisticRegression(
        max_iter=LOGREG_MAX_ITER,
        solver="liblinear",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", logreg),
        ]
    )

    print("Fitting Logistic Regression pipeline...")
    pipe.fit(X_train, y_train)

    # 9) Métrica de validación
    val_pred = pipe.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    print(f"\nValidation ROC AUC (Logistic Regression): {val_auc:.6f}")

    # 10) Coeficientes más influyentes
    try:
        feature_names = pipe.named_steps["prep"].get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(pipe.named_steps["prep"].transform(X_train[:1]).shape[1])]

    coefs = pipe.named_steps["clf"].coef_.ravel()
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs, "abs_coef": abs(coefs)})
    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    print("\nTop 20 features by |coeficiente|:")
    print(coef_df.head(20).to_string(index=False))

    # 11) Predicciones finales
    test_proba = pipe.predict_proba(X_test_to_predict_features)[:, 1]
    out = pd.DataFrame({"obs_id": test_obs_ids, "pred_proba": test_proba})
    out_path = Path("modelo_logistica.csv")
    out.to_csv(out_path, index=False)
    print(f"\n→ Predictions written to '{out_path.as_posix()}'")

    print("\n=== LOGISTIC REGRESSION pipeline complete ===")
    print(f"Tiempo total: {time.time() - t0:.2f} s")

if __name__ == "__main__":
    main()
