import json
import boto3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import logging
import os
from scipy.spatial.distance import cosine

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

def retrieve_similar_profiles(query_embedding, table_name, top_k=3):
    """Retrieve top-k similar profiles from DynamoDB using cosine similarity."""
    table = dynamodb.Table(table_name)
    response = table.scan()
    items = response['Items']
    
    similarities = []
    for item in items:
        stored_embedding = np.array(item['embedding'])
        similarity = 1 - cosine(query_embedding, stored_embedding)
        similarities.append((item['userId'], item['aboutMe'], similarity))
    
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:top_k]

def lambda_handler(event, context):
    """Lambda handler for RAG pipeline."""
    try:
        config = load_config()
        bucket = config['s3_bucket']
        data_prefix = config['s3_data_prefix']
        embeddings_table = config['dynamodb_embeddings_table']
        
        # Load input profile from event
        profile = json.loads(event['body'])
        about_me = profile['aboutMe']
        
        # Generate embedding for input profile
        embedding_model = SentenceTransformer(config['embedding_model'])
        query_embedding = embedding_model.encode(about_me, convert_to_numpy=True)
        
        # Retrieve similar profiles
        similar_profiles = retrieve_similar_profiles(query_embedding, embeddings_table)
        context = "\n".join([f"User {p[0]}: {p[1]} (Similarity: {p[2]:.2f})" for p in similar_profiles])
        
        # Load predictions
        response = s3_client.get_object(Bucket=bucket, Key=f"{data_prefix}predictions.csv")
        predictions_df = pd.read_csv(response['Body'])
        anomaly_score = predictions_df[predictions_df['userId'] == profile['userId']]['anomaly_score'].iloc[0]
        
        # Generate explanation with GPT-4o
        client = OpenAI(api_key=config['openai_api_key'])
        prompt = (
            f"You are an AI assistant for Scam Alert Detection. Given a user's profile and anomaly score, "
            f"explain why the profile might be flagged as a potential scam. Use the provided context of similar profiles. "
            f"Profile: {about_me}\nAnomaly Score: {anomaly_score:.2f}\nSimilar Profiles:\n{context}\n"
            f"Provide a concise explanation (100-150 words) with a friendly tone, using emojis 😊."
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=200
        )
        explanation = response.choices[0].message.content
        
        # Save explanation to S3
        result = {'userId': profile['userId'], 'anomaly_score': anomaly_score, 'explanation': explanation}
        with open('/tmp/result.json', 'w') as f:
            json.dump(result, f)
        s3_client.upload_file('/tmp/result.json', bucket, f"{data_prefix}results/{profile['userId']}.json")
        
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'RAG pipeline complete', 'result': result})
        }
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }