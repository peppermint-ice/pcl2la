import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from config import paths
import os

# Get paths
folder_paths = paths.get_paths()
combined_folder_path = folder_paths['combined']

# Load the provided CSV file

file_path = os.path.join(combined_folder_path,'all_plants.csv')
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
tsne_df_subset['leaf_area'] = subset_data['leaf_area']
tsne_df_subset['cultivar'] = subset_data['cultivar']
tsne_df_subset['treatment'] = subset_data['treatment']

# Visualize the results
plt.figure(figsize=(12, 8))

# Scatter plot
scatter = plt.scatter(tsne_df_subset['TSNE1'], tsne_df_subset['TSNE2'], c=tsne_df_subset['leaf_area'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Leaf Area')
plt.title('t-SNE visualization of plant data (subset)')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')

# Display the plot
plt.show()