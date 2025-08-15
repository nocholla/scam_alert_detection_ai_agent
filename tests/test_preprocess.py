import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
import os
from src.preprocess_data import preprocess_data
from src.data_loader import load_config

@pytest.fixture
def setup_data(tmpdir):
    config = {
        'data_dir': str(tmpdir.mkdir('data')),
        'tfidf_max_features': 100,
        'embedding_model': 'all-MiniLM-L6-v2'
    }
    profiles = pd.DataFrame({
        'userId': ['user_1', 'user_2'],
        'aboutMe': ['I love soccer', 'Looking for love'],
        'age': [25, 30],
        'country': ['Kenya', 'Nigeria'],
        'subscribed': [True, False],
        'relationshipGoals': ['Long-term', 'Casual']
    })
    profiles.to_csv(os.path.join(config['data_dir'], 'Profiles.csv'), index=False)
    return config, profiles

def test_preprocess_data(setup_data):
    config, profiles = setup_data
    processed_data, embeddings_df, tfidf, scaler, le_country, le_goals = preprocess_data(profiles, config)
    
    assert processed_data.shape[1] == 104  # 4 non-text features + 100 TF-IDF features
    assert embeddings_df.shape[1] == 385  # 384 embedding dimensions + userId
    assert os.path.exists(os.path.join(config['data_dir'], 'processed_data.csv'))
    assert os.path.exists(os.path.join(config['data_dir'], 'profile_embeddings.csv'))
    assert processed_data['userId'].tolist() == ['user_1', 'user_2']