import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
import os
import joblib
import torch
from src.predict import run_predictions
from src.data_loader import load_config
from src.lambda_predict import AnomalyNet

@pytest.fixture
def setup_data(tmpdir):
    config = {
        'data_dir': str(tmpdir.mkdir('data')),
        'models_dir': str(tmpdir.mkdir('models')),
        'tfidf_max_features': 100
    }
    profiles = pd.DataFrame({'userId': ['user_1'], 'aboutMe': ['Test profile']})
    processed_data = pd.DataFrame({'feature_0': [1.0], 'feature_1': [0.5]})
    profiles.to_csv(os.path.join(config['data_dir'], 'Profiles.csv'), index=False)
    processed_data.to_csv(os.path.join(config['data_dir'], 'processed_data.csv'), index=False)
    
    # Mock models
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    rf_model = RandomForestClassifier()
    xgb_model = XGBClassifier()
    rf_model.fit([[0, 1]], [0])
    xgb_model.fit([[0, 1]], [0])
    joblib.dump(rf_model, os.path.join(config['models_dir'], 'rf_model.pkl'))
    joblib.dump(xgb_model, os.path.join(config['models_dir'], 'xgb_model.pkl'))
    
    anomaly_net = AnomalyNet(input_dim=2)
    torch.save(anomaly_net.state_dict(), os.path.join(config['models_dir'], 'pytorch_model.pth'))
    
    return config

def test_run_predictions(setup_data):
    config = setup_data
    results = run_predictions(config)
    
    assert len(results) == 1
    assert 'userId' in results.columns
    assert 'anomaly_score' in results.columns
    assert results['userId'].iloc[0] == 'user_1'
    assert os.path.exists(os.path.join(config['data_dir'], 'predictions.csv'))