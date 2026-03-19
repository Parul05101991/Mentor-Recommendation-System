import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def recommend_mentors(mentee_info, mentor_csv, mentee_csv):
    """
    mentee_info: dict with keys 'InterestField', 'Location', 'BurnoutRiskScore'
    mentor_csv: path to mentors CSV
    mentee_csv: path to mentees CSV
    """
    # -----------------------------
    # Load Data
    # -----------------------------
    mentor_df = pd.read_csv(mentor_csv)
    mentee_df = pd.read_csv(mentee_csv, delimiter=';')

    # -----------------------------
    # Preprocess numeric burnout scores
    # -----------------------------
    mentor_df['Mentor_BurnoutRiskScore'] = mentor_df['Mentor_BurnoutRiskScore'].str.extract(r'(\d+)').astype(float)
    mentee_df['Mentee_BurnoutRiskScore'] = mentee_df['Mentee_BurnoutRiskScore'].str.extract(r'(\d+)').astype(float)

    # -----------------------------
    # One-hot encode categorical features
    # -----------------------------
    categorical_cols = ["InterestField", "Location"]
    
    # Fit encoder on mentor dataset, ignoring unknown categories later
    mentor_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    mentors_encoded = mentor_encoder.fit_transform(mentor_df[["Mentor_InterestField", "Mentor_Location"]])

    # Scale numeric burnout score
    scaler = StandardScaler()
    numerical_mentors_scaled = scaler.fit_transform(mentor_df[['Mentor_BurnoutRiskScore']])

    # Combine numeric + categorical features for mentors
    X_mentors = np.hstack([numerical_mentors_scaled, mentors_encoded])

    # -----------------------------
    # Encode the single input mentee
    # -----------------------------
    input_cat = mentor_encoder.transform([[mentee_info['InterestField'], mentee_info['Location']]])
    input_num = scaler.transform([[mentee_info['BurnoutRiskScore']]])
    input_vector = np.hstack([input_num, input_cat])

    # -----------------------------
    # Compute similarity with all mentors
    # -----------------------------
    similarity_scores = cosine_similarity(input_vector, X_mentors)[0]

    # -----------------------------
    # Get top 5 mentors
    # -----------------------------
    top_5_idx = np.argsort(similarity_scores)[-5:][::-1]
    mentor_list = []
    for idx in top_5_idx:
        mentor_list.append({
            'Mentor_ID': mentor_df.iloc[idx]['Mentor_ID'],
            'Mentor_InterestField': mentor_df.iloc[idx]['Mentor_InterestField'],
            'Mentor_Location': mentor_df.iloc[idx]['Mentor_Location'],
            'Similarity': similarity_scores[idx]
        })

    # We don't need clusters anymore for single input
    cluster = None

    return cluster, mentor_list