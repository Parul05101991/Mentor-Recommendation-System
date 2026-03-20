
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# -----------------------------
# 2. Create folders
# -----------------------------
os.makedirs("models", exist_ok=True)

# -----------------------------
# 3. Load Mentor Data
# -----------------------------
mentor_df = pd.read_csv('/Users/parulsharma/Documents/Mentor-Recommendation-System/mentors_database2.csv')
mentor_df = mentor_df.dropna()

# Extract numeric burnout score
mentor_df['Mentor_BurnoutRiskScore'] = mentor_df['Mentor_BurnoutRiskScore'] \
    .str.extract(r'(\d+)').astype(float)

# -----------------------------
# 4. Encode Mentor Features
# -----------------------------
categorical_mentors = ["Mentor_InterestField", "Mentor_Location"]

mentor_encoder = OneHotEncoder(sparse_output=False)
mentors_encoded = mentor_encoder.fit_transform(mentor_df[categorical_mentors])

numerical_mentors = ['Mentor_BurnoutRiskScore']
mentor_scaler = StandardScaler()
numerical_mentors_scaled = mentor_scaler.fit_transform(mentor_df[numerical_mentors])

X_mentors = np.hstack([numerical_mentors_scaled, mentors_encoded])

# -----------------------------
# 5. Load Mentee Data
# -----------------------------
mentee_df = pd.read_csv('/Users/parulsharma/Documents/Mentor-Recommendation-System/mentee_database.csv', delimiter=';')

mentee_df['Mentee_BurnoutRiskScore'] = mentee_df['Mentee_BurnoutRiskScore'] \
    .str.extract(r'(\d+)').astype(float)

# -----------------------------
# 6. Encode Mentee Features
# -----------------------------
categorical_mentees = ["Mentee_InterestField", "Mentee_Location"]

mentee_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
mentees_encoded = mentee_encoder.fit_transform(mentee_df[categorical_mentees])

numerical_mentees = ["Mentee_BurnoutRiskScore"]
mentee_scaler = StandardScaler()
numerical_mentees_scaled = mentee_scaler.fit_transform(mentee_df[numerical_mentees])

X_mentees = np.hstack([numerical_mentees_scaled, mentees_encoded])

# -----------------------------
# 7. Find Optimal Clusters
# -----------------------------
sil_scores = []
k_values = range(2, 61)  # reduced range for practicality

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_mentees)
    score = silhouette_score(X_mentees, labels)
    sil_scores.append(score)
    print(f'k={k}, Silhouette Score={score:.4f}')

best_k = k_values[np.argmax(sil_scores)]
print(f"\nBest number of clusters: {best_k}")

# Plot
plt.figure()
plt.plot(k_values, sil_scores, marker='o')
plt.title('Silhouette Score vs k')
plt.xlabel('k')
plt.ylabel('Score')
plt.grid()
plt.show()

# -----------------------------
# 8. Train Final KMeans
# -----------------------------
kmeans = KMeans(n_clusters=best_k, random_state=42)
mentee_df['Cluster'] = kmeans.fit_predict(X_mentees)

# -----------------------------
# 9. Compute Cluster Centroids
# -----------------------------
cluster_centroids = []
for cluster_id in range(best_k):
    points = X_mentees[mentee_df['Cluster'] == cluster_id]
    centroid = points.mean(axis=0)
    cluster_centroids.append(centroid)

cluster_centroids = np.array(cluster_centroids)

# -----------------------------
# 10. Compute Similarity
# -----------------------------
similarity_matrix = cosine_similarity(cluster_centroids, X_mentors)

# -----------------------------
# 11. Save Models
# -----------------------------
joblib.dump(mentor_encoder, 'models/mentor_encoder.pkl')
joblib.dump(mentee_encoder, 'models/mentee_encoder.pkl')
joblib.dump(mentor_scaler, 'models/mentor_scaler.pkl')
joblib.dump(mentee_scaler, 'models/mentee_scaler.pkl')
joblib.dump(kmeans, 'models/kmeans_model.pkl')

np.save('models/cluster_centroids.npy', cluster_centroids)
np.save('models/X_mentors.npy', X_mentors)

mentor_df.to_csv('models/mentor_df.csv', index=False)

print("\n✅ Training complete. Models saved in /models folder.")