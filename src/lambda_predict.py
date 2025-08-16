import json
import boto3
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

s3_client = boto3.client('s3')

class AnomalyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(AnomalyNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_model(model_path, bucket, input_dim=None):
    """Load model from S3."""
    s3_client.download_file(bucket, model_path, '/tmp/model')
    if model_path.endswith('.pkl'):
        return joblib.load('/tmp/model')
    elif model_path.endswith('.pth'):
        if input_dim is None:
            raise ValueError("input_dim is required for PyTorch models")
        model = AnomalyNet(input_dim=input_dim)
        model.load_state_dict(torch.load('/tmp/model', map_location=torch.device('cpu')))
        model.eval()
        return model
    return None

def lambda_handler(event, context):
    """Lambda handler for model predictions."""
    try:
        config = json.loads(s3_client.get_object(Bucket=os.environ['S3_BUCKET'], Key='config.yaml')['Body'].read())
        bucket = config['s3_bucket']
        data_prefix = config['s3_data_prefix']
        models_prefix = config['s3_models_prefix']
        input_dim = config['tfidf_max_features'] + 4  # TF-IDF + non-text features
        
        # Load processed data
        response = s3_client.get_object(Bucket=bucket, Key=f"{data_prefix}processed_data.csv")
        X = pd.read_csv(response['Body']).values
        
        # Load models
        rf_model = load_model(f"{models_prefix}rf_model.pkl", bucket)
        xgb_model = load_model(f"{models_prefix}xgb_model.pkl", bucket)
        anomaly_net = load_model(f"{models_prefix}pytorch_model.pth", bucket, input_dim)
        
        # Run predictions
        rf_preds = rf_model.predict_proba(X)[:, 1]
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
        results = [{'userId': f'user_{i}', 'anomaly_score': score} for i, score in enumerate(ensemble_scores)]
        
        # Save results to S3
        results_df = pd.DataFrame(results)
        results_df.to_csv('/tmp/predictions.csv', index=False)
        s3_client.upload_file('/tmp/predictions.csv', bucket, f"{data_prefix}predictions.csv")
        
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Predictions complete', 'results': results})
        }
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }