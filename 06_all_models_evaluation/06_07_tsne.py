import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from config import paths
import os
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np


# Get paths
folder_paths = paths.get_paths()
combined_folder_path = folder_paths['combined']

# Load the provided CSV file
file_path = os.path.join(combined_folder_path, 'all_plants.csv')
data = pd.read_csv(file_path)

# Data Preparation
# Encode categorical variables
data_encoded = data.copy()
label_encoders = {}
categorical_columns = ['cultivar', 'treatment']

for col in categorical_columns:
    le = LabelEncoder()
    data_encoded[col] = le.fit_transform(data_encoded[col])
    label_encoders[col] = le

# Select relevant features for t-SNE
features = ['exp', 'plant', 'cut', 'leaf_area'] + categorical_columns
X = data_encoded[features]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce the number of samples for t-SNE (use a subset of 200 samples)
subset_data = data_encoded.sample(n=200, random_state=42)

# Select relevant features for t-SNE from the subset
X_subset = subset_data[features]

# Normalize the data
X_subset_scaled = scaler.transform(X_subset)

# Run t-SNE with adjusted parameters for faster computation
tsne = TSNE(n_components=2, random_state=42, init='pca', n_iter=500)
X_tsne_subset = tsne.fit_transform(X_subset_scaled)

# Create a DataFrame with the t-SNE results and original labels
tsne_df_subset = pd.DataFrame(X_tsne_subset, columns=['TSNE1', 'TSNE2'])
for col in features:
    tsne_df_subset[col] = subset_data[col].values

tsne_df_subset = tsne_df_subset.dropna(subset=['cultivar'])

# Visualize the results
plt.figure(figsize=(12, 8))
scatter = plt.scatter(tsne_df_subset['TSNE1'], tsne_df_subset['TSNE2'], c=tsne_df_subset['leaf_area'], cmap='viridis', alpha=0.7, s=70)

# Add colorbar and customize its font size
colorbar = plt.colorbar(scatter)
colorbar.set_label('Measured TLA, cm²', fontsize=16)  # Set colorbar label font size
colorbar.ax.tick_params(labelsize=16)  # Set colorbar tick label font size

# Customize plot labels and title
plt.title('t-SNE visualization of TLA', fontsize=16)
plt.xlabel('TSNE1', fontsize=16)
plt.ylabel('TSNE2', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Show plot
# plt.show()
plt.savefig(os.path.join(folder_paths['plots'], '20250116', 'tsne.png'), dpi=600)


# Prepare the data for clustering analysis
X_tsne = tsne_df_subset[['TSNE1', 'TSNE2']]

# Perform K-Means clustering (assume 3 clusters for simplicity)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_tsne)

# Calculate the silhouette score
silhouette = silhouette_score(X_tsne, kmeans_labels)

# Evaluate clustering purity for all categorical labels
aris = {}
for col in features:
    if col in categorical_columns or col == 'plant' or col == 'exp' or col == 'cut':  # Include relevant labels
        aris[col] = adjusted_rand_score(tsne_df_subset[col], kmeans_labels)

# Trustworthiness metric: proportion of neighbors preserved in t-SNE
# Align X_subset_scaled with the cleaned subset
X_subset_scaled = scaler.transform(tsne_df_subset[features])

# Use k=5 nearest neighbors
k_neighbors = 5
nbrs_original = NearestNeighbors(n_neighbors=k_neighbors).fit(X_subset_scaled)
nbrs_tsne = NearestNeighbors(n_neighbors=k_neighbors).fit(X_tsne)

# Get neighbors for the subset in the original and t-SNE space
distances_orig, indices_orig = nbrs_original.kneighbors(X_subset_scaled)
distances_tsne, indices_tsne = nbrs_tsne.kneighbors(X_tsne)

# Initialize variables
n = len(X_subset_scaled)
trustworthiness_score = 0

# Trustworthiness calculation
for i in range(n):
    for j in indices_tsne[i]:
        match = np.where(indices_orig[i] == j)[0]
        if match.size > 0:  # If there's a match, calculate the penalty
            rank = match[0]
            penalty = max(0, rank - k_neighbors + 1)
            trustworthiness_score += penalty

# Normalize the trustworthiness score
trustworthiness = 1 - (trustworthiness_score * 2 / (n * k_neighbors * (2 * n - 3 * k_neighbors - 1)))

# Correlation between TSNE components and all features
correlations = {}
for col in features:
    correlations[f"Correlation (TSNE1 and {col})"] = tsne_df_subset['TSNE1'].corr(tsne_df_subset[col])
    correlations[f"Correlation (TSNE2 and {col})"] = tsne_df_subset['TSNE2'].corr(tsne_df_subset[col])

# Print results
print("Silhouette Score:", silhouette)
print("Adjusted Rand Indexes:")
for col, ari in aris.items():
    print(f"  {col}: {ari}")
print("Trustworthiness:", trustworthiness)
print("Correlations with TSNE components:")
for key, value in correlations.items():
    print(f"  {key}: {value}")