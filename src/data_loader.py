import pandas as pd
import yaml
import logging
import joblib
import torch
import os
from src.lambda_predict import AnomalyNet  # Import AnomalyNet for PyTorch model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file {config_path} not found")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        raise

def load_data(data_path):
    """Load data from CSV file."""
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path}")
        return df
    except FileNotFoundError:
        logger.error(f"Data file {data_path} not found")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Data file {data_path} is empty")
        raise

def load_model(model_path, input_dim=None):
    """Load a model from file, supporting scikit-learn and PyTorch models."""
    try:
        if model_path.endswith('.pkl'):
            model = joblib.load(model_path)
            logger.info(f"Loaded scikit-learn model from {model_path}")
            return model
        elif model_path.endswith('.pth'):
            if input_dim is None:
                raise ValueError("input_dim is required for loading PyTorch models")
            model = AnomalyNet(input_dim=input_dim)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            logger.info(f"Loaded PyTorch model from {model_path}")
            return model
        else:
            raise ValueError(f"Unsupported model file format: {model_path}")
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise