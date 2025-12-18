# customer_pipeline.py
"""
Shared ML pipeline components for the customer project.

This module holds all custom transformers and helper functions that are used
both in training and in inference (FastAPI app), so that joblib pickles
refer to a stable module path: `customer_pipeline.<name>`.
"""


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



# =============================================================================
# Building blocks for preprocessing
# =============================================================================


cat_pipeline = make_pipeline(
    OneHotEncoder(handle_unknown="ignore")
)

num_pipeline = make_pipeline(
    StandardScaler()
)

def build_preprocessing():
    """
    Return the ColumnTransformer preprocessing used in the customer models.
    """
    car_attribs = ['hasCrCard', 'isActiveMember', 'gender', 'geography', 'isZeroBalance']
    num_attribs = ['creditScore', 'age', 'tenure', 'balance', 'numofProducts', 'estimatedSalary']
    preprocessing = ColumnTransformer(
        [
            ("num",  num_pipeline, num_attribs),
            ("cat",  cat_pipeline, car_attribs),
        ],
        remainder='drop',
    )
    return preprocessing


# =============================================================================
# Estimator factory used by both non-Optuna and Optuna code
# =============================================================================

def make_estimator_for_name(name: str):
    """
    Given a model name, return an unconfigured estimator instance.
    Used in PCA variants and (optionally) elsewhere.
    """
    if name == "ridge":
        return RidgeClassifier()
    elif name == "histgradientboosting":
        return HistGradientBoostingClassifier(random_state=42)
    elif name == "xgboost":
        return XGBClassifier(
            objective="reg:squarederror",
            random_state=42,
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
        )
    elif name == "lightgbm":
        return LGBMClassifier(
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")