"""
Phase 3: Business Logic & Interpretability Layer
"""

import pandas as pd
import numpy as np
import config
import matplotlib.pyplot as plt
import shap
from src.interest_funcs import calculate_risk_premium, get_break_even_rate
from src.model import CreditFeatureEngineer


def get_individual_prediction(model_pipeline, input_df: pd.DataFrame):
    """
    Passes raw data through the pipeline to get a calibrated PD.
    """
    # The pipeline handles the Feature Engineering + Encoding internally
    pd_value = model_pipeline.predict_proba(input_df)[0, 1]
    return pd_value


def get_individual_expected_profit(PD, r, ead, lgd):
    ep = ((1-PD) * r * ead) - (PD * lgd * ead)
    return ep


def get_shap_explanation(model_pipeline, input_df: pd.DataFrame):
    """
    Explains a single prediction by reaching into the Pipeline steps.
    """
    # 1. Extract the steps from the pipeline
    actual_pipeline = model_pipeline.estimator
    feature_engineer = actual_pipeline.named_steps['feature_engineering']
    preprocessor = actual_pipeline.named_steps['preprocessing']
    model = actual_pipeline.named_steps['model']

    # 2. Manually transform the data to get the final feature names/values
    engineered_df = feature_engineer.transform(input_df)
    processed_X = preprocessor.transform(engineered_df)
    
    # 3. Get feature names from the column transformer
    raw_feature_names = preprocessor.get_feature_names_out()
    clean_feature_names = [name.split('__')[-1] for name in raw_feature_names]

    # 4. Create Explainer (XGBoost specific)
    explainer = shap.TreeExplainer(model, feature_names=clean_feature_names)
    shap_values = explainer(processed_X)
    
    # 5. Plotting
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("Risk Driver Analysis")
    plt.tight_layout()
    
    return fig


def generate_risk_report(pd_val, ead, lgd, prime_rate):
    """
    Calculates the financial metrics for a single client.
    """
    premium = calculate_risk_premium(pd_val)
    interest_rate = prime_rate + premium
    
    # EP = EAD * r * (1-PD) - EAD * LGD * PD
    expected_profit = (ead * interest_rate * (1 - pd_val)) - (ead * lgd * pd_val)
    
    break_even = get_break_even_rate(pd_val, lgd)
    
    return {
        "PD": pd_val,
        "Premium": premium,
        "Total Rate": interest_rate,
        "Expected Profit": expected_profit,
        "Break Even Rate": break_even,
        "Risk Category": "High" if pd_val > 0.20 else "Standard"
    }


# Test if PD and EP predictions work
if __name__ == "__main__":
    import joblib
    pipeline = joblib.load(config.get_model_path())
    
    test_client = pd.DataFrame({
        'Age': [33], 'Sex': ['male'], 'Job': ['skilled'], 'Housing': ['own'],
        'Saving accounts': ['little'], 'Checking account': ['moderate'],
        'Credit amount': [5000], 'Duration': [12], 'Purpose': ['car']
    })
    
    pd_result = get_individual_prediction(pipeline, test_client)
    report = generate_risk_report(pd_result, 5000, 0.45, 0.1025)
    print(f"Client PD: {report['PD']:.2%}")
    print(f"Expected Profit: R{report['Expected Profit']:.2f}")