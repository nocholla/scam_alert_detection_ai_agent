import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import logging
import os
from src.data_loader import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_ml_models():
    config = load_config()
    data_dir = config['data_dir']
    models_dir = config['models_dir']
    tfidf_max_features = config['tfidf_max_features']
    
    # Load data
    df = pd.read_csv(os.path.join(data_dir, 'Profiles.csv'))
    
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
    
    # Combine features
    X = np.hstack((non_text_features, aboutMe_tfidf))
    # Dummy labels (replace with actual labels if available)
    y = np.random.randint(0, 2, size=len(df))  # Simulated binary labels (0=normal, 1=anomaly)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    joblib.dump(rf_model, os.path.join(models_dir, 'rf_model.pkl'))
    logger.info("Random Forest model saved")
    
    # Train XGBoost
    from xgboost import XGBClassifier
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X, y)
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgb_model.pkl'))
    logger.info("XGBoost model saved")

if __name__ == "__main__":
    train_ml_models()
