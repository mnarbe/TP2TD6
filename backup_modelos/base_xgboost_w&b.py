import constants as C
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK
import wandb  # <--- Nuevo import


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
        "random_state": C.RAND_SEED,
        "bootstrap": True,
    }
    rf_params = default_params.copy()
    if params:
        rf_params.update(params)

    model = RandomForestClassifier(**rf_params)

    print("  → Fitting RandomForestClassifier...")
    model.fit(X_train, y_train)
    print("  → Model training complete.")
    return model

def train_classifier_xgboost(X_train, y_train, params=None):
    """
    Train a Classifier 
    """

    model = xgb.XGBClassifier(objective = 'binary:logistic',
                                seed = C.RAND_SEED,
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
                                eval_metric = 'auc',
                                enable_categorical=True,
                                **params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)]
    )
    return model

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
    mean_auc = np.mean(aucs)
    
    # Loguear el AUC y los hiperparámetros en wandb
    wandb.log({**params, "mean_auc": mean_auc})
    
    return {"loss": 1 - mean_auc, "status": STATUS_OK}


def trainXGBoostModel(X_train, y_train, fold_splits, optimization_evals):
    # Define hyperparameter search space
    space = {
        'max_depth': hp.uniformint('max_depth', 3, 30),
        'gamma': hp.uniform('gamma', 0, 4),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
        'reg_lambda': hp.uniform('reg_lambda', 0, 4),
        'reg_alpha': hp.uniform('reg_alpha', 0, 4),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'n_estimators': hp.uniformint('n_estimators', 10, 150),
        'min_child_weight': hp.uniform('min_child_weight', 1, 10)
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

    # Loggear hiperparámetros en wandb
    wandb.config.update(params)

    # Entrenar modelo final
    model = train_classifier_xgboost(X_train, y_train, params)

    # Loggear métrica de entrenamiento (ejemplo: AUC)
    preds = model.predict_proba(X_train)[:, 1]
    auc = roc_auc_score(y_train, preds)
    wandb.log({"train_auc_final": auc})

    return model