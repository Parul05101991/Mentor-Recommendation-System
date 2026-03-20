
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_models():
    """Load trained models, encoders, scalers, and mentor data."""
    mentor_encoder = joblib.load('/Users/parulsharma/Documents/Mentor-Recommendation-System/models/mentor_encoder.pkl')
    mentee_encoder = joblib.load('/Users/parulsharma/Documents/Mentor-Recommendation-System/models/mentee_encoder.pkl')
    mentor_scaler = joblib.load('/Users/parulsharma/Documents/Mentor-Recommendation-System/models/mentor_scaler.pkl')
    mentee_scaler = joblib.load('/Users/parulsharma/Documents/Mentor-Recommendation-System/models/mentee_scaler.pkl')
    kmeans = joblib.load('/Users/parulsharma/Documents/Mentor-Recommendation-System/models/kmeans_model.pkl')

    cluster_centroids = np.load('/Users/parulsharma/Documents/Mentor-Recommendation-System/models/cluster_centroids.npy')
    X_mentors = np.load('/Users/parulsharma/Documents/Mentor-Recommendation-System/models/X_mentors.npy')
    
    # Load saved mentor dataframe (consistent with training)
    mentor_df = pd.read_csv('/Users/parulsharma/Documents/Mentor-Recommendation-System/models/mentor_df.csv')
    
    return mentor_encoder, mentee_encoder, mentor_scaler, mentee_scaler, kmeans, cluster_centroids, X_mentors, mentor_df


def recommend_mentors(models, mentee_input, top_k=5):
    """
    Recommend top mentors for a new mentee.

    models: tuple of loaded models and data
    mentee_input: dict with 'InterestField', 'Location', 'BurnoutRiskScore'
    top_k: number of top mentors to return
    """
    mentor_encoder, mentee_encoder, mentor_scaler, mentee_scaler, kmeans, cluster_centroids, X_mentors, mentor_df = models

    # Encode categorical features
    mentee_cat = pd.DataFrame([{
        'Mentee_InterestField': mentee_input['InterestField'],
        'Mentee_Location': mentee_input['Location']
    }])
    mentee_encoded = mentee_encoder.transform(mentee_cat)

    # Scale numerical features
    mentee_num = pd.DataFrame([{'Mentee_BurnoutRiskScore': mentee_input['BurnoutRiskScore']}])
    mentee_scaled = mentee_scaler.transform(mentee_num)

    # Combine features
    X_mentee = np.hstack([mentee_scaled, mentee_encoded])

    # Predict cluster
    cluster = kmeans.predict(X_mentee)[0]

    # Compute similarity with mentors
    similarity = cosine_similarity([cluster_centroids[cluster]], X_mentors)[0]
    top_indices = similarity.argsort()[-top_k:][::-1]

    mentor_list = []
    for idx in top_indices:
        mentor_info = mentor_df.iloc[idx].to_dict()
        mentor_info['Similarity'] = similarity[idx]
        mentor_list.append(mentor_info)

    return cluster, mentor_list