import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from config import paths

def load_scaler(scaler_file_path):
    with open(scaler_file_path, 'rb') as file:
        scaler = pickle.load(file)
    return scaler

def inverse_transform(scaled_values, mean, scale):
    return scaled_values * scale + mean

# Model parameters
algorithm_name = "alpha"
parameter_value = "3"
assessment_name = "trimesh"
dataset_type = "simple"
elimination_status = "elim"
regression_model = "en"
byyear = "noyear"

# Set folder paths
folder_paths = paths.get_paths()
models_folder_path = folder_paths["models"]
global_test_sets_path = folder_paths["global_test_sets"]
scalers_folder_path = folder_paths["scalers"]

# Load the model
model_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_{elimination_status}_best_model_{regression_model}_{byyear}.pkl"
model_file_path = os.path.join(models_folder_path, model_file_name)

with open(model_file_path, 'rb') as file:
    model = pickle.load(file)
print("Model loaded")

# Load the scaler
scaler_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_scaler.pkl"
scaler_file_path = os.path.join(scalers_folder_path, scaler_file_name)
scaler = load_scaler(scaler_file_path)
print("Scaler loaded")

# Load the global test set
global_test_set_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_{elimination_status}_{regression_model}_global_test_set_{byyear}.csv"
global_test_set_file_path = os.path.join(global_test_sets_path, global_test_set_file_name)
global_test_df = pd.read_csv(global_test_set_file_path)

# Extract features and target
X_test = global_test_df.drop(columns=['measured_leaf_area'])
y_test = global_test_df['measured_leaf_area']

# Reverse scaling for 'measured_leaf_area'
y_test_unscaled = inverse_transform(y_test, scaler.mean_[-1], scaler.scale_[-1])

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(y_test_unscaled, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Measured Leaf Area (unscaled)')
plt.ylabel('Frequency')
plt.title('Histogram of Measured Leaf Area (Unscaled)')
plt.grid(True)
plt.show()
