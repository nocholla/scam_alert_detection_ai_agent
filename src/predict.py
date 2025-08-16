import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import joblib
import torch
import logging
import os
from src.data_loader import load_config, load_data, load_model
from src.lambda_predict import AnomalyNet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_predictions(config):
    """Run predictions using Random Forest, XGBoost, and AnomalyNet."""
    try:
        # Load data
        data_dir = config['data_dir']
        processed_data_path = os.path.join(data_dir, 'processed_data.csv')
        profiles_path = os.path.join(data_dir, 'Profiles.csv')
        if not os.path.exists(processed_data_path) or not os.path.exists(profiles_path):
            raise FileNotFoundError("Processed data or profiles not found")
        
        processed_data = pd.read_csv(processed_data_path)
        profiles_df = pd.read_csv(profiles_path)
        X = processed_data[[col for col in processed_data.columns if col.startswith('feature_')]].values
        
        # Load models
        models_dir = config['models_dir']
        rf_model = load_model(os.path.join(models_dir, 'rf_model.pkl'))
        xgb_model = load_model(os.path.join(models_dir, 'xgb_model.pkl'))
        anomaly_net = load_model(os.path.join(models_dir, 'pytorch_model.pth'), input_dim=X.shape[1])
        
        # Run predictions
        rf_preds = rf_model.predict_proba(X)[:, 1]  # Probability of anomaly
        xgb_preds = xgb_model.predict_proba(X)[:, 1]
        anomaly_scores = []
        for x in X:
            x_tensor = torch.FloatTensor(x).unsqueeze(0)
            with torch.no_grad():
                reconstructed = anomaly_net(x_tensor)
                mse = torch.mean((reconstructed - x_tensor) ** 2).item()
                anomaly_scores.append(mse)
        
        # Ensemble predictions
        ensemble_scores = (rf_preds + xgb_preds + np.array(anomaly_scores)) / 3
        results = pd.DataFrame({
            'userId': profiles_df['userId'],
            'anomaly_score': ensemble_scores
        })
        
        # Save results
        results_path = os.path.join(data_dir, 'predictions.csv')
        results.to_csv(results_path, index=False)
        logger.info(f"Saved predictions to {results_path}")
        
        return results
    except Exception as e:
        logger.error(f"Error in predictions: {e}")
        raise

if __name__ == "__main__":
    config = load_config()
    results = run_predictions(config)
    logger.info("Predictions completed successfully")