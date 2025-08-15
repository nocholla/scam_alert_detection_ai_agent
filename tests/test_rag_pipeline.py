import pytest
import boto3
import json
from moto import mock_s3, mock_dynamodb
from src.rag_pipeline import lambda_handler

@mock_s3
@mock_dynamodb
def test_rag_pipeline():
    # Setup mock S3 and DynamoDB
    s3_client = boto3.client('s3')
    dynamodb = boto3.resource('dynamodb')
    bucket = 'scam-alert-detection-models'
    s3_client.create_bucket(Bucket=bucket)
    
    # Mock config
    config = {
        's3_bucket': bucket,
        's3_data_prefix': 'data/',
        'dynamodb_embeddings_table': 'ProfileEmbeddings',
        'embedding_model': 'all-MiniLM-L6-v2',
        'openai_api_key': 'dummy_key'
    }
    s3_client.put_object(Bucket=bucket, Key='config.yaml', Body=json.dumps(config))
    
    # Mock predictions
    predictions_df = pd.DataFrame([{'userId': 'user_1', 'anomaly_score': 0.8}])
    predictions_df.to_csv('/tmp/predictions.csv', index=False)
    s3_client.upload_file('/tmp/predictions.csv', bucket, 'data/predictions.csv')
    
    # Mock DynamoDB
    table = dynamodb.create_table(
        TableName='ProfileEmbeddings',
        KeySchema=[{'AttributeName': 'userId', 'KeyType': 'HASH'}],
        AttributeDefinitions=[{'AttributeName': 'userId', 'AttributeType': 'S'}],
        BillingMode='PAY_PER_REQUEST'
    )
    table.put_item(Item={
        'userId': 'user_2',
        'embedding': [0.1] * 384,
        'aboutMe': 'Looking for love'
    })
    
    # Test Lambda handler
    event = {'body': json.dumps({'userId': 'user_1', 'aboutMe': 'Hi I love soccer'})}
    response = lambda_handler(event, None)
    
    assert response['statusCode'] == 200
    assert 'result' in json.loads(response['body'])
    assert json.loads(response['body'])['result']['userId'] == 'user_1'
