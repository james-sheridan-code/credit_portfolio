"""
Phase 1: Model Training & Pipeline Construction
"""

import config # use 'python -m src.model' to run this
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Any
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


def main():
    # 1. Load and split data
    X_train, X_test, y_train, y_test = load_and_split_data(config.get_data_path())
    
    # 2. Build untrained pipelines
    xgb_pipeline = build_pipeline('xgb')
    lr_pipeline = build_pipeline('lr')

    # 3. Train, Evaluate, and Calibrate
    calibrated_xgb = evaluate_and_calibrate(xgb_pipeline, X_train, y_train, X_test, y_test, "XGBoost")
    calibrated_lr = evaluate_and_calibrate(lr_pipeline, X_train, y_train, X_test, y_test, "Logistic Regression")

    # 4. Saving the calibrated XGBoost Classifier for later usage (only XGBC as it is the superior model)
    joblib.dump(calibrated_xgb, config.get_model_path())
    print("Production pipeline saved (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧ 'production_xgb_pipeline.pkl'")


def load_and_split_data(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads raw data, formats target, and splits into train/test."""
    df = pd.read_csv(filepath)
    
    if "index" in df.columns:
        df = df.drop("index", axis=1)

    # Clean the target variable
    df['Risk'] = df['Risk'].replace({'good': 0, 'bad': 1}).astype(int)
    
    # Fill NaNs 
    df['Saving accounts'] = df['Saving accounts'].fillna('little')
    df['Checking account'] = df['Checking account'].fillna('little')

    X = df.drop(columns=['Risk'])
    y = df['Risk']

    return train_test_split(X, y, test_size=0.30, random_state=0, stratify=y)


def build_pipeline(model_type: str = 'xgb') -> Pipeline:
    """
    Constructs the full Pipeline: Engineering -> Preprocessing -> Model.
    """
    # Define feature groups
    nominal_features = ['Housing', 'Purpose', 'Sex', 'Job']
    ordinal_features = ['Saving accounts', 'Checking account']
    numeric_features = ['Age', 'Duration', 'Credit amount', 
                        'credit_per_age', 'credit_per_duration', 'age_duration_ratio']

    # Ordinal categories (ordered from worst to best)
    saving_cats = ['little', 'moderate', 'quite rich', 'rich']
    checking_cats = ['little', 'moderate', 'rich']

    # Create the ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('Numeric features', StandardScaler(), numeric_features),
            ('Nominal features', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_features),
            ('Ordinal features', OrdinalEncoder(categories=[saving_cats, checking_cats], 
                                   handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features)],
        remainder='drop')

    # Select the model
    if model_type == 'xgb':
        classifier = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
            reg_lambda=20, gamma=2, eval_metric='logloss', random_state=0)
    else:
        classifier = LogisticRegression(max_iter=10000, C=0.01, random_state=0)

    # Assemble the final pipeline
    pipeline = Pipeline(steps=[
        ('feature_engineering', CreditFeatureEngineer()),
        ('preprocessing', preprocessor),
        ('model', classifier)
    ])

    return pipeline


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering class for the pipeline.
    """
    def fit(self, X, y=None):
        return self # Nothing to fit for mathematical transformations
        
    def transform(self, X):
        X_copy = X.copy()
        
        epsilon = 1e-6 # no division by zero
        
        X_copy["credit_per_age"] = X_copy["Credit amount"] / (X_copy["Age"] + epsilon)
        X_copy["credit_per_duration"] = X_copy["Credit amount"] / (X_copy["Duration"] + epsilon)
        X_copy["age_duration_ratio"] = X_copy["Age"] / (X_copy["Duration"] + epsilon)
        
        return X_copy


def evaluate_and_calibrate(pipeline: Pipeline, X_train, y_train, X_test, y_test, name: str) -> Pipeline:
    """Trains, evaluates, calibrates, and returns the calibrated pipeline."""
    print(f"--- Training {name} ---")
    pipeline.fit(X_train, y_train)
    
    # Raw Evaluation
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    print(f"Initial ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Initial Brier Score: {brier_score_loss(y_test, y_prob):.4f}")

    # Calibration (Wraps the entire fitted pipeline)
    print("Calibrating model...")
    calibrated_pipeline = CalibratedClassifierCV(pipeline, method='sigmoid', cv=10)
    calibrated_pipeline.fit(X_test, y_test) # Fit calibration on holdout/test set
    
    cali_probs = calibrated_pipeline.predict_proba(X_test)[:, 1]
    print(f"Calibrated Brier Score: {brier_score_loss(y_test, cali_probs):.4f}\n")
    
    return calibrated_pipeline


if __name__ == '__main__':
    main()