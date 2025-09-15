import constants as C

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK

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

    print("  --> Fitting RandomForestClassifier...")
    model.fit(X_train, y_train)
    print("  --> Model training complete.")
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
    return {"loss": 1 - np.mean(aucs), "status": STATUS_OK}

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
    from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    import numpy as np
    import pandas as pd
    import constants as C
    
    print("=== Training XGBoost with Temporal Validation ===")
    
    # Define hyperparameter space
    space = {
        'max_depth': hp.uniformint('max_depth', 3, 15),
        'gamma': hp.uniform('gamma', 0, 5),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'reg_lambda': hp.uniform('reg_lambda', 0, 10),
        'reg_alpha': hp.uniform('reg_alpha', 0, 10),
        'subsample': hp.uniform('subsample', 0.6, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.6, 1),
        'n_estimators': hp.uniformint('n_estimators', 100, 1000),
        'min_child_weight': hp.uniformint('min_child_weight', 1, 15)
    }
    
    def objective_temporal(params):
        """Objective function for hyperopt using temporal validation."""
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            seed=C.RAND_SEED,
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
    
    # Run hyperparameter optimization
    print(f"Starting hyperparameter optimization with {max_evals} evaluations...")
    best = fmin(
        fn=objective_temporal,
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
    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        seed=C.RAND_SEED,
        eval_metric='auc',
        enable_categorical=True,
        **best_params
    )
    
    # Train only on pre-2024 data (more conservative approach)
    final_model.fit(X_train, y_train)
    
    # Evaluate final model
    train_preds = final_model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_preds)
    
    val_preds = final_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_preds)
    
    print(f"\nFinal Model Performance:")
    print(f"  Training AUC (pre-2024): {train_auc:.5f}")
    print(f"  Validation AUC (2024): {val_auc:.5f}")
    
    # Feature importance
    if hasattr(final_model, 'feature_importances_'):
        importances = final_model.feature_importances_
        feature_names = final_model.get_booster().feature_names
        imp_series = pd.Series(importances, index=feature_names)
        imp_sorted = imp_series.sort_values(ascending=False)
        
        print("\nTop 10 feature importances:")
        print(imp_sorted.head(10))
    
    return final_model