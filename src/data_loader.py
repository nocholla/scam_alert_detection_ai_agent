import pandas as pd
import yaml
import boto3
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

s3_client = boto3.client('s3')

def load_config():
    """Load configuration from S3 or local file."""
    try:
        if 'S3_BUCKET' in os.environ:
            bucket = os.environ['S3_BUCKET']
            response = s3_client.get_object(Bucket=bucket, Key='config.yaml')
            config = yaml.safe_load(response['Body'].read())
        else:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

def load_data(data_dir="data", s3_bucket=None, s3_data_prefix=None):
    """Load datasets from local directory or S3."""
    datasets = {}
    file_names = ['Profiles.csv', 'BlockedUsers.csv', 'DeclinedUsers.csv', 'DeletedUsers.csv', 'ReportedUsers.csv']
    
    try:
        if s3_bucket and s3_data_prefix:
            for file_name in file_names:
                s3_key = f"{s3_data_prefix}{file_name}"
                response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
                datasets[file_name.split('.')[0]] = pd.read_csv(response['Body'])
                logger.info(f"Loaded {file_name} from S3")
        else:
            for file_name in file_names:
                file_path = os.path.join(data_dir, file_name)
                if os.path.exists(file_path):
                    datasets[file_name.split('.')[0]] = pd.read_csv(file_path)
                    logger.info(f"Loaded {file_name} from local directory")
                else:
                    logger.warning(f"{file_name} not found in local directory")
        return datasets
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def load_model(model_path, s3_bucket=None):
    """Load model from local directory or S3."""
    import joblib
    import torch
    try:
        if s3_bucket:
            s3_client.download_file(s3_bucket, model_path, '/tmp/model')
            model_path = '/tmp/model'
        if model_path.endswith('.pkl'):
            model = joblib.load(model_path)
        elif model_path.endswith('.pth'):
            model = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            raise ValueError("Unsupported model file format")
        logger.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise