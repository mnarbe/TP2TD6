import constants as C

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, ParameterGrid
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK
from tqdm import tqdm

USES_GPU = False # Colocar en True si tenes una GPU de NVIDIA
DEVICE = 'cuda' if USES_GPU else 'cpu'

def train_classifier_xgboost(X_train, y_train, params=None):
    """
    Train a Classifier 
    """

    model = xgb.XGBClassifier(objective = 'binary:logistic',
                                seed = C.RAND_SEED,
                                device=DEVICE,
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
                                seed = C.RAND_SEED,
                                device=DEVICE,
                                eval_metric = 'auc',
                                enable_categorical=True,
                                **params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)]
    )
    return model

################################################
########      Funciones Objetivo      ##########
################################################

def objective_KFold(params, X_train, y_train, fold_splits):
    """
    CV interna sobre el train inicial (sin agrupar por usuario).
    """
    kf = KFold(n_splits=fold_splits, shuffle=True, random_state=C.RAND_SEED)
    aucs = []
    for tr_idx, va_idx in kf.split(X_train, y_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        model = train_classifier_xgboost_val(X_tr, y_tr, X_va, y_va, params)
        preds = model.predict_proba(X_va)[:, 1]
        aucs.append(roc_auc_score(y_va, preds))
    return {"loss": 1 - np.mean(aucs), "status": STATUS_OK}

def objective_temporal(params, X_train, y_train, X_val, y_val):
    """Objective function for hyperopt using temporal validation."""
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        seed=C.RAND_SEED,
        device=DEVICE,
        eval_metric='auc',
        enable_categorical=True,
        **params
    )
    
    # Train on pre-2024 data, validate on 2024 data
    model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        verbose=False # Stop if no improvement for 50 rounds
    )
    
    # Get predictions on validation set
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    
    print(f"  --> Temporal Validation AUC: {auc:.5f}")
    
    return {"loss": 1 - auc, "status": STATUS_OK}

def objective_temporal_por_usuario(params, X_train, y_train, X_val, y_val,
                                   username_col="username", n_splits=5):
    """
    Validación cruzada por usuario sobre un split temporal fijo.
    En cada fold se entrena sólo con los usuarios que aparecen en ese fold de validación.
    Esto simula el escenario productivo donde predecimos para un conjunto fijo de usuarios.
    """
    usuarios_unicos = X_val[username_col].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=C.RAND_SEED_2)

    auc_scores = []

    for fold, (train_users_idx, val_users_idx) in enumerate(kf.split(usuarios_unicos)):
        val_users_fold = usuarios_unicos[val_users_idx]

        # --- Train: sólo historial de los usuarios que aparecen en este fold de validación ---
        train_mask_users = X_train[username_col].isin(val_users_fold)
        X_train_fold = X_train[train_mask_users]
        y_train_fold = y_train[train_mask_users]

        # --- Valid: datos futuros de los mismos usuarios ---
        val_mask = X_val[username_col].isin(val_users_fold)
        X_val_fold = X_val[val_mask]
        y_val_fold = y_val[val_mask]

        # Entrenar modelo
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            seed=C.RAND_SEED,
            device=DEVICE,
            eval_metric='auc',
            enable_categorical=True,
            **params
        )

        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )

        preds = model.predict_proba(X_val_fold)[:, 1]
        auc = roc_auc_score(y_val_fold, preds)
        auc_scores.append(auc)

        print(f"Fold {fold+1}/{n_splits} - AUC: {auc:.5f} - train rows: {len(X_train_fold)} - val rows: {len(X_val_fold)}")

    mean_auc = np.mean(auc_scores)
    print(f"--> AUC promedio (entrenando con los mismos usuarios del fold): {mean_auc:.5f}")

    return {"loss": 1 - mean_auc, "status": STATUS_OK}

################################################
########      Feature Selection      ###########
################################################

def backward_feature_selection_topN(
    X_train, y_train,
    X_val, y_val,
    min_features=15,
    topN=30,
    topN_used=8,
    xgb_params=None,
    early_stopping_rounds=50,
    verbose=True,
    save_path="resultados/topN_features.csv"
):
    """
    Realiza backward feature elimination con XGBoost usando 'gain' como criterio de importancia.
    Guarda las mejores N combinaciones de features con su AUC y orden de importancia.
    """

    if xgb_params is None:
        xgb_params = {
            "n_estimators": 220,
            "max_depth": 5,
            "learning_rate": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_lambda": 3,
            "reg_alpha": 1,
            "min_child_weight": 10
        }

    features = list(X_train.columns)
    history = []

    pbar = tqdm(total=len(features) - min_features, disable=not verbose)
    last_removed = None

    while len(features) >= min_features:
        # Entrenar modelo
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            seed=C.RAND_SEED_3,
            device=DEVICE,
            eval_metric='auc',
            enable_categorical=True,
            early_stopping_rounds=early_stopping_rounds,
            **xgb_params
        )
        model.fit(
            X_train[features], y_train,
            eval_set=[(X_val[features], y_val)],
            verbose=False
        )

        # Importancias
        booster = model.get_booster()
        imp = booster.get_score(importance_type="gain")
        imp_series = pd.Series({f: imp.get(f, 0.0) for f in features}).sort_values(ascending=False)

        # Guardar estado actual
        history.append({
            "n_features": len(features),
            "val_auc": model.best_score,
            "features": features.copy(),
            "importances": imp_series.to_dict(),
            "removed_feature": last_removed
        })

        # Cortar si llegamos al mínimo de features
        if len(features) == min_features:
            break

        # Feature menos importante
        least_important = imp_series.idxmin()

        # Eliminarla y guardar cuál fue
        last_removed = least_important
        features.remove(least_important)

        pbar.update(1)

    pbar.close()

    # Convertir historial a DataFrame
    history_df = pd.DataFrame(history)

    # Tomar las top N combinaciones según AUC
    topN_df = history_df.sort_values("val_auc", ascending=False).head(topN).reset_index(drop=True)

    if save_path is not None:
        topN_df.to_csv(save_path, index=False)
        if verbose:
            print(f"✅ Top {topN} combinaciones guardadas en: {save_path}")

    best_rows = topN_df.head(topN_used)
    best_features_list = best_rows['features'].tolist()

    return best_features_list

def backward_feature_selection_por_user_topN(
    X_train, y_train,
    X_val, y_val,
    username_col="username",
    n_splits=5,
    min_features=15,
    topN=30,
    topN_used=8,
    xgb_params=None,
    early_stopping_rounds=50,
    verbose=True,
    save_path="resultados/topN_features_grouped_cv.csv",
    importance_type="gain",   # "gain" o "perm" si lo cambias después
    fold_weight_by_size=False
):
    """
    Backward feature elimination promediando importancias entre folds (agrupado por usuario).
    Versión optimizada: los folds y subconjuntos de datos por usuario se precomputan una vez.
    """

    if xgb_params is None:
        xgb_params = {
            "n_estimators": 220,
            "max_depth": 5,
            "learning_rate": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_lambda": 3,
            "reg_alpha": 1,
            "min_child_weight": 10
        }

    # --- 1️⃣ Folds fijos y subconjuntos precomputados ---
    usuarios_unicos = X_val[username_col].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=C.RAND_SEED_2)
    folds = list(kf.split(usuarios_unicos))

    # Precomputar los DataFrames para cada fold
    fold_data = []
    for fold, (train_users_idx, val_users_idx) in enumerate(folds):
        val_users_fold = usuarios_unicos[val_users_idx]

        train_mask_users = X_train[username_col].isin(val_users_fold)
        val_mask = X_val[username_col].isin(val_users_fold)

        X_train_fold = X_train.loc[train_mask_users].reset_index(drop=True)
        y_train_fold = y_train.loc[train_mask_users].reset_index(drop=True)
        X_val_fold = X_val.loc[val_mask].reset_index(drop=True)
        y_val_fold = y_val.loc[val_mask].reset_index(drop=True)

        fold_data.append({
            "val_users": val_users_fold,
            "X_train": X_train_fold,
            "y_train": y_train_fold,
            "X_val": X_val_fold,
            "y_val": y_val_fold
        })

        if verbose:
            print(f"✅ Fold {fold+1}: train={len(X_train_fold)}, val={len(X_val_fold)}, users={len(val_users_fold)}")

    # --- 2️⃣ Backward loop ---
    features = list(X_train.columns)
    history = []
    total_iterations = max(0, len(features) - min_features)
    pbar = tqdm(total=total_iterations, disable=not verbose)

    while len(features) >= min_features:
        fold_importances = []
        fold_aucs = []
        fold_sizes = []

        for fold, fd in enumerate(fold_data):
            X_train_fold = fd["X_train"][features]
            y_train_fold = fd["y_train"]
            X_val_fold = fd["X_val"][features]
            y_val_fold = fd["y_val"]

            if len(X_train_fold) == 0 or len(X_val_fold) == 0:
                continue

            model = xgb.XGBClassifier(
                objective='binary:logistic',
                seed=C.RAND_SEED,
                device=DEVICE,
                eval_metric='auc',
                enable_categorical=True,
                early_stopping_rounds=early_stopping_rounds,
                **xgb_params
            )

            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )

            preds = model.predict_proba(X_val_fold)[:, 1]
            auc = roc_auc_score(y_val_fold, preds)
            fold_aucs.append(auc)
            fold_sizes.append(len(X_val_fold))

            # Importancia tipo "gain"
            booster = model.get_booster()
            imp = booster.get_score(importance_type=importance_type)
            imp_series = pd.Series({f: imp.get(f, 0.0) for f in features})
            fold_importances.append(imp_series)

            if verbose:
                print(f"Iter {len(features)}f - Fold {fold+1}/{n_splits} - AUC={auc:.5f}")

        # --- Calcular importancias promedio ---
        if len(fold_importances) == 0:
            print("No se pudieron calcular importancias en ningún fold. Terminando loop.")
            break

        weights = (
            np.array(fold_sizes) / np.sum(fold_sizes)
            if fold_weight_by_size else
            np.ones(len(fold_importances)) / len(fold_importances)
        )

        imp_df = pd.concat(fold_importances, axis=1).fillna(0.0)
        weighted_mean = (imp_df * weights).sum(axis=1)
        weighted_std = np.sqrt(((imp_df - weighted_mean[:, None])**2 * weights).sum(axis=1))
        freq_nonzero = (imp_df > 0).sum(axis=1) / imp_df.shape[1]

        imp_stats = pd.DataFrame({
            "mean_imp": weighted_mean,
            "std_imp": weighted_std,
            "freq_nonzero": freq_nonzero
        }).sort_values("mean_imp", ascending=False)

        mean_auc = np.mean(fold_aucs)
        history.append({
            "n_features": len(features),
            "val_auc": mean_auc,
            "features": features.copy(),
            "importances_mean": imp_stats["mean_imp"].to_dict(),
            "importances_std": imp_stats["std_imp"].to_dict(),
            "importances_freq": imp_stats["freq_nonzero"].to_dict()
        })

        if verbose:
            print(f"--> Iteración con {len(features)} features - AUC promedio: {mean_auc:.5f}")
            print("   Menor importancia:", imp_stats.tail(1).index[0])

        # Corte si se alcanzó el mínimo
        if len(features) == min_features:
            break

        # Eliminar feature menos importante (menor mean gain)
        least_important = imp_stats["mean_imp"].idxmin()
        features.remove(least_important)
        pbar.update(1)

    pbar.close()

    # --- 3️⃣ Resultados finales ---
    history_df = pd.DataFrame(history)
    topN_df = history_df.sort_values("val_auc", ascending=False).head(topN).reset_index(drop=True)

    if save_path is not None:
        topN_df.to_csv(save_path, index=False)
        if verbose:
            print(f"✅ Top {topN} combinaciones guardadas en: {save_path}")

    best_rows = topN_df.head(topN_used)
    best_features_list = best_rows['features'].tolist()

    return best_features_list

################################################
#########      Modelos XGBoost      ############
################################################

def trainXGBoostModel(X_train, y_train, fold_splits, optimization_evals):
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

    # Optimización de hiperparametros
    best = fmin(
        fn=lambda params: objective_KFold(params, X_train, pd.Series(y_train), fold_splits),
        space=space,
        algo=tpe.suggest,
        max_evals=optimization_evals,
        rstate=np.random.default_rng(C.RAND_SEED_2),
    )
    
    # Guardamos los hiperparámetros ganadores.
    params = space_eval(space, best)
    print("\nBest hyperparameters found:")
    print(params)

    # Entrenar modelo final
    return train_classifier_xgboost(X_train, y_train, params)

def trainXGBoostModel_v2(X_train, y_train, fold_splits, optimization_evals):
    '''
    Medio mala, por el momento ignorar un poco.
    '''
    from sklearn.model_selection import GridSearchCV, ParameterGrid
    import xgboost as xgb
    import numpy as np

    # Estimador base (categóricas habilitadas para dtype 'category')
    base_estimator = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        device=DEVICE,
        enable_categorical=True,
        random_state=C.RAND_SEED,
        n_estimators=300,
    )

    # Espacios de búsqueda (reducí listas para menos combinaciones)
    param_grid = {
        "max_depth": [3, 5, 7],                # antes: [3,4,5,6,7,8]
        "min_child_weight": [1, 3, 5],         # antes: [1,2,3,4,5,6]
        "gamma": [0.0, 0.5, 1.0],              # antes: [0.0,0.1,0.5,1.0,2.0]
        "learning_rate": [0.02, 0.05, 0.1],    # antes: [0.01,0.02,0.05,0.08,0.1,0.15,0.2]
        "subsample": [0.7, 1.0],               # antes: [0.6,0.7,0.8,0.9,1.0]
        "colsample_bytree": [0.7, 1.0],        # antes: [0.6,0.7,0.8,0.9,1.0]
        "reg_alpha": [0.0, 0.5, 1.0],          # igual
        "reg_lambda": [0.5, 1.0, 2.0],         # antes: [0.0,0.5,1.0,2.0]
        "n_estimators": [200, 400],            # antes: [100,200,300,400,500]
    }

    # Armamos todas las combinaciones y aplicamos un SUBCONJUNTO aleatorio
    all_params = list(ParameterGrid(param_grid))
    total_full = len(all_params)

    # Usamos optimization_evals como LÍMITE de combinaciones a evaluar
    max_combos = int(optimization_evals) if int(optimization_evals) > 0 else total_full
    if max_combos < total_full:
        rng = np.random.RandomState(C.RAND_SEED)
        idx = rng.choice(total_full, size=max_combos, replace=False)
        all_params = [all_params[i] for i in idx]

    total = len(all_params)
    print(f"GridSearch con límite: {total}/{total_full} combinaciones a evaluar...\n")

    search = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,   # se ignora en _run_search (usamos 'all_params')
        scoring="roc_auc",
        cv=fold_splits,
        n_jobs=-1,               # paralelismo; no afecta cantidad de combinaciones
        verbose=0,
        refit=False,
        error_score="raise"
    )

    # Monkeypatch para recorrer SOLO 'all_params' y mostrar progreso + AUC
    def run_search_with_progress(evaluate_candidates):
        for i, params in enumerate(all_params, start=1):
            print(f"[{i}/{total}] Probando: {params}")
            evaluate_candidates([params])
            try:
                cvr = search.cv_results_
                mean_auc = float(cvr["mean_test_score"][-1])
                std_auc = float(cvr["std_test_score"][-1])
                print(f"   --> AUC CV: {mean_auc:.5f} ± {std_auc:.5f}\n")
            except Exception:
                print("   --> AUC CV: (no disponible aún)\n")

    search._run_search = run_search_with_progress  # type: ignore

    search.fit(X_train, y_train)

    best_params = search.best_params_
    print("\nMejores hiperparámetros (GridSearchCV):")
    print(best_params)

    return train_classifier_xgboost(X_train, y_train, best_params)



# Para validation 2024 y train pre-2024
def trainXGBoostModelTemporal(X_train, y_train, X_val, y_val, max_evals):
    """
    Train XGBoost model using temporal validation (no cross-validation).
    Uses pre-2024 data for training and 2024 data for validation.
    
    Args:
        X_train: Training features (pre-2024)
        y_train: Training target (pre-2024)  
        X_val: Validation features (2024)
        y_val: Validation target (2024)
        max_evals: Number of hyperparameter evaluations
    
    Returns:
        Trained XGBoost model
    """
    
    print("=== Training XGBoost with Temporal Validation ===")
    
    # Define hyperparameter space
    space = {
        'max_depth': hp.uniformint('max_depth', 3, 8),
        'gamma': hp.uniform('gamma', 0, 5),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.1)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(0.01), np.log(10)),
        'reg_alpha':  hp.loguniform('reg_alpha',  np.log(0.01), np.log(10)),
        'subsample': hp.uniform('subsample', 0.6, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 0.8),
        'colsample_bylevel': 0.7,
        'n_estimators': hp.uniformint('n_estimators', 200, 600),
        'min_child_weight': hp.uniformint('min_child_weight', 10, 80)
    }
    
    # Run hyperparameter optimization
    print(f"Starting hyperparameter optimization with {max_evals} evaluations...")
    best = fmin(
        fn=lambda params: objective_temporal(params, X_train, y_train, X_val, y_val),
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        rstate=np.random.default_rng(C.RAND_SEED_2) if hasattr(C, 'RAND_SEED_2') else np.random.default_rng(42),
        verbose=False
    )
    
    # Get best parameters
    best_params = space_eval(space, best)
    print("\nBest hyperparameters found:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Train final model with best parameters on all available training data
    print("\nTraining final model with best parameters...")
    final_model = train_classifier_xgboost(X_train, y_train, best_params)
    
    return final_model

def trainXGBoostModelTemporalPorUser(X_train, y_train, X_val, y_val, max_evals):
    """
    Train XGBoost model using temporal validation (user cross-validation).
    Uses pre-2024 data for training and 2024 data for validation.
    
    Args:
        X_train: Training features (pre-2024)
        y_train: Training target (pre-2024)  
        X_val: Validation features (2024)
        y_val: Validation target (2024)
        max_evals: Number of hyperparameter evaluations
    
    Returns:
        Trained XGBoost model
    """
    
    print("=== Training XGBoost with Temporal Validation ===")
    
    # Define hyperparameter space
    space = {
        'max_depth': hp.uniformint('max_depth', 3, 8),
        'gamma': hp.uniform('gamma', 0, 4),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.1)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(0.01), np.log(10)),
        'reg_alpha':  hp.loguniform('reg_alpha',  np.log(0.01), np.log(10)),
        'subsample': hp.uniform('subsample', 0.6, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 0.8),
        'colsample_bylevel': 0.7,
        'n_estimators': hp.uniformint('n_estimators', 200, 600),
        'min_child_weight': hp.uniformint('min_child_weight', 10, 80)
    }
    
    # Run hyperparameter optimization
    print(f"Starting hyperparameter optimization with {max_evals} evaluations...")
    best = fmin(
        fn=lambda params: objective_temporal_por_usuario(params, X_train, y_train, X_val, y_val),
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        rstate=np.random.default_rng(C.RAND_SEED_2) if hasattr(C, 'RAND_SEED_2') else np.random.default_rng(42),
        verbose=False
    )
    
    # Get best parameters
    best_params = space_eval(space, best)
    print("\nBest hyperparameters found:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Convert to series
    y_train = pd.Series(y_train)
    y_val = pd.Series(y_val)

    # Use all data to train
    X_train_final = pd.concat([X_train, X_val], axis=0)
    y_train_final = pd.concat([y_train, y_val], axis=0)
    
    # Train final model with best parameters on all available training data
    print("\nTraining final model with best parameters and all data...")
    final_model = train_classifier_xgboost(X_train_final, y_train_final, best_params)
    
    return final_model