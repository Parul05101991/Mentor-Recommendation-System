import streamlit as st
import pandas as pd

# Import the function from mentors_connect.py
from mentor_connect import recommend_mentors

# Load mentee dataframe for dropdowns
mentee_df = pd.read_csv(
    '/Users/parulsharma/Documents/Techlabs Project/Mentor-Recommendation-System/mentee_database.csv',
    delimiter=';'
)

# UI: select interest and location
interest = st.selectbox('Select Interest Field', mentee_df['Mentee_InterestField'].unique())
location = st.selectbox('Select Location', mentee_df['Mentee_Location'].unique())

# UI: slider for burnout risk
burnout = st.slider('Burnout Risk Score', 0.0, 10.0, 3.0)

# Button: call the recommend_mentors function
if st.button('Find Mentors'):
    cluster, mentor_list = recommend_mentors(
        {'InterestField': interest, 'Location': location, 'BurnoutRiskScore': burnout},
        '/Users/parulsharma/Documents/Techlabs Project/Mentor-Recommendation-System/mentors_database2.csv',
        '/Users/parulsharma/Documents/Techlabs Project/Mentor-Recommendation-System/mentee_database.csv'
    )

    # Display top mentors
    st.write('Top recommended mentors:')
    for mentor in mentor_list:
        st.write(f"Mentor ID: {mentor['Mentor_ID']}, Field: {mentor['Mentor_InterestField']}, "
                 f"Location: {mentor['Mentor_Location']}, Similarity: {mentor['Similarity']:.4f}")

