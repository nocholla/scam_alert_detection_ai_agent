import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import os
from moto import mock_s3
import boto3
import yaml
from src.data_loader import load_config, load_data, load_model

@pytest.fixture
def setup_s3():
    with mock_s3():
        s3_client = boto3.client('s3')
        bucket = 'scam-alert-detection-models'
        s3_client.create_bucket(Bucket=bucket)
        config = {
            'data_dir': 'data',
            'models_dir': 'models',
            's3_bucket': bucket,
            's3_data_prefix': 'data/',
            's3_models_prefix': 'models/'
        }
        s3_client.put_object(Bucket=bucket, Key='config.yaml', Body=yaml.dump(config))
        os.environ['S3_BUCKET'] = bucket
        yield s3_client, bucket

def test_load_config(setup_s3):
    s3_client, bucket = setup_s3
    config = load_config()
    assert config['s3_bucket'] == bucket
    assert config['data_dir'] == 'data'

def test_load_data_local(tmpdir):
    # Create mock data
    data_dir = tmpdir.mkdir('data')
    profiles = pd.DataFrame({'userId': ['user_1'], 'aboutMe': ['Test profile']})
    profiles.to_csv(data_dir.join('Profiles.csv'), index=False)
    
    datasets = load_data(data_dir=str(data_dir))
    assert 'Profiles' in datasets
    assert len(datasets['Profiles']) == 1
    assert datasets['Profiles']['userId'].iloc[0] == 'user_1'

def test_load_data_s3(setup_s3):
    s3_client, bucket = setup_s3
    profiles = pd.DataFrame({'userId': ['user_1'], 'aboutMe': ['Test profile']})
    profiles.to_csv('/tmp/Profiles.csv', index=False)
    s3_client.upload_file('/tmp/Profiles.csv', bucket, 'data/Profiles.csv')
    
    datasets = load_data(s3_bucket=bucket, s3_data_prefix='data/')
    assert 'Profiles' in datasets
    assert len(datasets['Profiles']) == 1
    assert datasets['Profiles']['userId'].iloc[0] == 'user_1'