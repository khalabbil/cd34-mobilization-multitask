"""
src/preprocessing.py
═══════════════════════════════════════════════════════════════════════
Fold-internal imputer + scaler. Manuscript-parity: RobustScaler.
Imputation upgraded to IterativeImputer (vs manuscript median).
═══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
from sklearn.experimental import enable_iterative_imputer    # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import RobustScaler


def fit_preprocessor(X_train: np.ndarray, cfg: dict):
    imputer_kind = cfg["preprocess"]["imputer"]
    if imputer_kind == "iterative":
        imp = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=cfg["preprocess"]["imputer_max_iter"],
            random_state=cfg["run"]["seed"],
        )
    else:
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy="median")
    imp.fit(X_train)
    X_tr = imp.transform(X_train)
    scaler_kind = cfg["preprocess"]["scaler"]
    scaler = RobustScaler() if scaler_kind == "robust" else None
    if scaler is not None:
        scaler.fit(X_tr)
        X_tr = scaler.transform(X_tr)
    return imp, scaler, X_tr


def apply_preprocessor(X: np.ndarray, imp, scaler) -> np.ndarray:
    X_ = imp.transform(X)
    if scaler is not None:
        X_ = scaler.transform(X_)
    return X_
