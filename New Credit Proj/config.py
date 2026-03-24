from pathlib import Path
import joblib


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Ensure folders exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

def get_model_path():
    """Returns the path to the model pipeline."""
    return MODELS_DIR / "production_xgb_pipeline.pkl"

def get_data_path():
    """Returns the path to the main CSV."""
    return DATA_DIR / "credit_data.csv"