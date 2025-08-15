import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import logging
import os
from src.data_loader import load_config, load_data
from src.lambda_predict import AnomalyNet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_anomaly_net(config):
    """Train AnomalyNet model and save to models directory."""
    try:
        # Load data
        data_dir = config['data_dir']
        processed_data_path = os.path.join(data_dir, 'processed_data.csv')
        if not os.path.exists(processed_data_path):
            raise FileNotFoundError(f"Processed data not found at {processed_data_path}")
        
        processed_data = pd.read_csv(processed_data_path)
        X = processed_data[[col for col in processed_data.columns if col.startswith('feature_')]].values
        
        # Initialize model
        input_dim = X.shape[1]
        model = AnomalyNet(input_dim=input_dim, hidden_dim=64)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Convert data to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Training loop
        epochs = config['anomaly_epochs']
        batch_size = config['batch_size']
        num_batches = len(X_tensor) // batch_size
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i + batch_size]
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Save model
        models_dir = config['models_dir']
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, 'pytorch_model.pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved AnomalyNet model to {model_path}")
        
        return model
    except Exception as e:
        logger.error(f"Error training AnomalyNet: {e}")
        raise

if __name__ == "__main__":
    config = load_config()
    train_anomaly_net(config)