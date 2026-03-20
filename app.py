# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mentee_recommender import load_models, recommend_mentors

# -----------------------------
# 1. Load trained models
# -----------------------------
models = load_models()

# -----------------------------
# 2. Streamlit UI
# -----------------------------
st.title("Mentee-Mentor Matching App")

# Dropdown options for UI
# You can hardcode or fetch from mentor data
mentor_df = models[-1]  # mentor_df
interest_fields = sorted(mentor_df['Mentor_InterestField'].unique())
locations = sorted(mentor_df['Mentor_Location'].unique())

interest = st.selectbox('Select Interest Field', interest_fields)
location = st.selectbox('Select Location', locations)
burnout = st.slider('Burnout Risk Score', 0.0, 10.0, 3.0)

# -----------------------------
# 3. Recommend mentors
# -----------------------------
if st.button('Find Mentors'):
    cluster, mentor_list = recommend_mentors(models, {
        'InterestField': interest,
        'Location': location,
        'BurnoutRiskScore': burnout
    })

    st.write(f'✅ Mentee belongs to cluster {cluster}')
    st.write('### Top recommended mentors:')
    for mentor in mentor_list:
        st.write(
            f"Mentor ID: {mentor['Mentor_ID']}, "
            f"Field: {mentor['Mentor_InterestField']}, "
            f"Location: {mentor['Mentor_Location']}, "
            f"Similarity: {mentor['Similarity']:.4f}"
        )

    # -----------------------------
    # 4. Optional heatmap
    # -----------------------------
    # Checkbox for heatmap
if st.checkbox("Show heatmap for top mentors"):
    # Make sure mentor_list exists and is not empty
    if 'mentor_list' in locals() and len(mentor_list) > 0:
        # Get similarity values and mentor IDs
        similarity_values = [m['Similarity'] for m in mentor_list]
        mentor_ids = [str(m['Mentor_ID']) for m in mentor_list]

        # Create explicit figure
        fig, ax = plt.subplots(figsize=(8, 2))
        sns.heatmap([similarity_values],
                    annot=True,
                    yticklabels=[f'Cluster {cluster}'],
                    xticklabels=mentor_ids,
                    cmap='YlGnBu',
                    cbar=True,
                    ax=ax)
        plt.title("Similarity Heatmap")
        st.pyplot(fig)
    else:
        st.warning("Please press 'Find Mentors' first to generate a heatmap.")