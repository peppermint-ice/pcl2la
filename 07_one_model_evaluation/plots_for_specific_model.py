import pickle
import xgboost as xgb
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             mean_absolute_percentage_error, median_absolute_error, explained_variance_score)
import numpy as np
from config import paths
from datetime import datetime
from sklearn.decomposition import PCA

def inverse_transform(scaled_values, mean, scale):
    return scaled_values * scale + mean


model_file_name = "marching_cubes_13_trimesh_repaired_noElim_best_model_xgb_noyear.json"

# Set the pattern to use regex
pattern = r"(\w*_?\w*?)_(\w*[.]?\w*)_(\w*)_(\w*)_(\w*)_best_model_(\w*)_(\w*)\.(\w*)"
match = re.search(pattern, model_file_name)
# Get parameters of the file
if match:
    algorithm_name = match.group(1)
    parameter_value = match.group(2)
    assessment_name = match.group(3)
    dataset_type = match.group(4)
    elimination_status = match.group(5)
    regression_model = match.group(6)
    byyear = match.group(7)
    file_type = match.group(8)
print(algorithm_name, parameter_value, assessment_name, dataset_type, elimination_status, regression_model, file_type)

# Set folder paths
folder_paths = paths.get_paths()
models_folder_path = folder_paths["models"]
global_test_sets_path = folder_paths["global_test_sets"]
train_folder_path = folder_paths["train_sets"]
test_folder_path = folder_paths["test_sets"]
ready_for_training_path = folder_paths["ready_for_training"]

# Load a model
model_file_path = os.path.join(models_folder_path, model_file_name)
if regression_model == 'xgb':
    model = xgb.Booster()
    model.load_model(model_file_path)
else:
    with open(model_file_path, 'rb') as file:
        model = pickle.load(file)
print("Model loaded")

# Load test set
global_test_set_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_{elimination_status}_global_test_set_{byyear}.csv"
global_test_set_file_path = os.path.join(global_test_sets_path, global_test_set_file_name)
global_test_df = pd.read_csv(global_test_set_file_path)
print(len(global_test_df))

# Extract features and target
X_test = global_test_df.drop(columns=['measured_leaf_area'])
y_test = global_test_df['measured_leaf_area']

# Convert the test set into DMatrix, which is the data structure used by XGBoost
dtest = xgb.DMatrix(X_test)

# Make predictions
y_pred = model.predict(dtest)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
median_ae = median_absolute_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)

# Print performance metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"Median Absolute Error: {median_ae}")
print(f"Explained Variance Score: {evs}")

# Function to format text
def format_text(text):
    return text.replace('_', ' ').title()

# Format algorithm_name and parameter_value for display
formatted_algorithm_name = format_text(algorithm_name)
formatted_parameter_value = format_text(parameter_value)

parameter_value_type = {
    'marching_cubes': 'Threshold',
    'poisson': 'Depth',
    'alpha': 'Alpha Value',
    'ball_pivoting': 'Smallest Ball Radius'
}

# Plot actual vs. predicted values with a 1:1 line and metrics
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Measured Leaf Area, Cm²')
plt.ylabel('Predicted Leaf Area, Cm²')
plt.title('Actual Vs. Predicted Measured Leaf Area')
plt.axis('square')
plt.xlim([min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())])
plt.ylim([min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())])
plt.text(0.05, 0.95, f'R² = {r2:.2f}\nRMSE = {rmse:.2f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.text(0.05, 0.85, f'3D Reconstruction Algorithm: {formatted_algorithm_name}\n{parameter_value_type[algorithm_name]} Value: {formatted_parameter_value}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.show()

# Plot feature importance
importance = model.get_score(importance_type='weight')
importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
features = [format_text(item[0]) for item in importance]
scores = [item[1] for item in importance]

plt.figure(figsize=(12, 6))
plt.barh(features, scores)
plt.xlabel('Importance Score')
plt.title('Feature Importance')
plt.show()

# Apply PCA and plot principal components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_test)

pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['measured_leaf_area'] = y_test.values

plt.figure(figsize=(12, 6))
plt.scatter(pca_df['Principal Component 1'], pca_df['Principal Component 2'], c=pca_df['measured_leaf_area'], cmap='viridis', alpha=0.6)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Pca Of Test Set')
plt.colorbar(label='Measured Leaf Area')
plt.show()
