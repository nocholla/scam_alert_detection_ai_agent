import json
import boto3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

def load_config():
    """Load configuration from S3."""
    bucket = os.environ['S3_BUCKET']
    response = s3_client.get_object(Bucket=bucket, Key='config.yaml')
    import yaml
    return yaml.safe_load(response['Body'].read())

def lambda_handler(event, context):
    """Lambda handler for preprocessing profile data."""
    try:
        config = load_config()
        tfidf_max_features = config['tfidf_max_features']
        bucket = config['s3_bucket']
        data_prefix = config['s3_data_prefix']
        
        # Load profile data from S3
        response = s3_client.get_object(Bucket=bucket, Key=f"{data_prefix}Profiles.csv")
        df = pd.read_csv(response['Body'])
        
        # Preprocess non-text features
        le_country = LabelEncoder()
        le_goals = LabelEncoder()
        scaler = StandardScaler()
        df['country_encoded'] = le_country.fit_transform(df['country'])
        df['relationshipGoals_encoded'] = le_goals.fit_transform(df['relationshipGoals'])
        df['subscribed'] = df['subscribed'].astype(float)
        non_text_features = scaler.fit_transform(df[['age', 'country_encoded', 'subscribed', 'relationshipGoals_encoded']])
        
        # TF-IDF for aboutMe
        tfidf = TfidfVectorizer(max_features=tfidf_max_features)
        aboutMe_tfidf = tfidf.fit_transform(df['aboutMe']).toarray()
        logger.info(f"Fitted TfidfVectorizer with {tfidf_max_features} features")
        
        # Sentence Transformer embeddings
        embedding_model = SentenceTransformer(config['embedding_model'])
        embeddings = embedding_model.encode(df['aboutMe'].tolist(), convert_to_numpy=True)
        
        # Combine features
        X = np.hstack((non_text_features, aboutMe_tfidf))
        
        # Save to S3
        processed_data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        processed_data.to_csv('/tmp/processed_data.csv', index=False)
        s3_client.upload_file('/tmp/processed_data.csv', bucket, f"{data_prefix}processed_data.csv")
        
        # Save embeddings to DynamoDB
        table = dynamodb.Table(config['dynamodb_embeddings_table'])
        for idx, (user_id, embedding) in enumerate(zip(df['userId'], embeddings)):
            table.put_item(Item={
                'userId': user_id,
                'embedding': embedding.tolist(),
                'aboutMe': df['aboutMe'][idx]
            })
        
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Preprocessing complete', 'feature_count': X.shape[1]})
        }
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }