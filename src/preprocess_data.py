import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import logging
import os
from src.data_loader import load_config, load_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data(profiles_df, config):
    """Preprocess profile data for model training or inference."""
    try:
        tfidf_max_features = config['tfidf_max_features']
        embedding_model_name = config['embedding_model']
        
        # Preprocess non-text features
        le_country = LabelEncoder()
        le_goals = LabelEncoder()
        scaler = StandardScaler()
        
        profiles_df['country_encoded'] = le_country.fit_transform(profiles_df['country'])
        profiles_df['relationshipGoals_encoded'] = le_goals.fit_transform(profiles_df['relationshipGoals'])
        profiles_df['subscribed'] = profiles_df['subscribed'].astype(float)
        non_text_features = scaler.fit_transform(
            profiles_df[['age', 'country_encoded', 'subscribed', 'relationshipGoals_encoded']]
        )
        
        # TF-IDF for aboutMe
        tfidf = TfidfVectorizer(max_features=tfidf_max_features)
        aboutMe_tfidf = tfidf.fit_transform(profiles_df['aboutMe']).toarray()
        logger.info(f"Fitted TfidfVectorizer with {tfidf_max_features} features")
        
        # Sentence Transformer embeddings
        embedding_model = SentenceTransformer(embedding_model_name)
        embeddings = embedding_model.encode(profiles_df['aboutMe'].tolist(), convert_to_numpy=True)
        
        # Combine features
        X = np.hstack((non_text_features, aboutMe_tfidf))
        
        # Save processed data and embeddings
        processed_data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        processed_data['userId'] = profiles_df['userId']
        embeddings_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])
        embeddings_df['userId'] = profiles_df['userId']
        
        output_dir = config['data_dir']
        processed_data.to_csv(os.path.join(output_dir, 'processed_data.csv'), index=False)
        embeddings_df.to_csv(os.path.join(output_dir, 'profile_embeddings.csv'), index=False)
        logger.info(f"Saved processed data and embeddings to {output_dir}")
        
        return processed_data, embeddings_df, tfidf, scaler, le_country, le_goals
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

if __name__ == "__main__":
    config = load_config()
    datasets = load_data(config['data_dir'])
    profiles_df = datasets.get('Profiles', pd.DataFrame())
    if not profiles_df.empty:
        processed_data, embeddings_df, _, _, _, _ = preprocess_data(profiles_df, config)
        logger.info("Preprocessing completed successfully")
    else:
        logger.error("No profile data available for preprocessing")