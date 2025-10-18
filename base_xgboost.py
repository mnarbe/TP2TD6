import constants as C

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK
from tqdm import tqdm

USES_GPU = False # Colocar en True para GPU de NVIDIA
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
    early_stopping_rounds=30,
    verbose=True,
    save_path="resultados/topN_features.csv"
):
    """
    Realiza backward feature elimination con XGBoost usando 'gain' como criterio de importancia.
    Guarda las mejores N combinaciones de features con su AUC y orden de importancia.
    """

    if xgb_params is None:
        xgb_params = {
            "n_estimators": 180,
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
            print(f" Top {topN} combinaciones guardadas en: {save_path}")

    best_rows = topN_df.head(topN_used)
    best_features_list = best_rows['features'].tolist()

    return best_features_list

################################################
#########      Modelos XGBoost      ############
################################################

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
