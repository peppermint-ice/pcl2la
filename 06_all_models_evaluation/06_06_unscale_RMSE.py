import pickle
import xgboost as xgb
import os
import re
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from config import paths


def load_scaler(scaler_file_path):
    with open(scaler_file_path, 'rb') as file:
        scaler = pickle.load(file)
    return scaler


def inverse_transform(scaled_values, mean, scale):
    return scaled_values * scale + mean


def run_any_model(regression_model, x, model):
    # Run the model. XGBoost requires specific approach
    if regression_model == 'xgb':
        dmatrix = xgb.DMatrix(x)
        y_pred = model.predict(dmatrix)
    else:
        y_pred = model.predict(x)
    return y_pred


# Set folder paths
folder_paths = paths.get_paths()
models_folder_path = folder_paths["models"]
global_test_sets_path = folder_paths["global_test_sets"]
scalers_folder_path = folder_paths["scalers"]
combined_folder_path = folder_paths["combined"]

# Load the results CSV file
results_file_path = os.path.join(combined_folder_path, "combine_ML_240724_1608.csv")
results_df = pd.read_csv(results_file_path)

# Iterate over each row in the results DataFrame
for index, row in results_df.iterrows():
    algorithm_name = row['algorithm_name']
    parameter_value = row['parameter_value']
    assessment_name = row['assessment_name']
    dataset_type = row['dataset_type']
    elimination_status = row['elimination_status']
    regression_model = row['regression_model']
    byyear = row['byyear']

    if float(parameter_value).is_integer():
        parameter_value = int(parameter_value)

    # Set the model file path
    if regression_model == 'xgb':
        model_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_{elimination_status}_best_model_{regression_model}_{byyear}.json"
    else:
        model_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_{elimination_status}_best_model_{regression_model}_{byyear}.pkl"

    model_file_path = os.path.join(models_folder_path, model_file_name)

    # Load the model
    if regression_model == 'xgb':
        model = xgb.Booster()
        model.load_model(model_file_path)
    else:
        with open(model_file_path, 'rb') as file:
            model = pickle.load(file)

    print(f"Model loaded for {model_file_name}")

    # Load the scaler if elimination status is "elim"
    scaler = None
    if elimination_status == "elim":
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

    # Make predictions
    y_pred = run_any_model(regression_model, X_test, model)

    # Reverse scaling if elimination status is "elim"
    if elimination_status == "elim" and scaler is not None:
        y_pred = inverse_transform(y_pred, scaler.mean_[-1], scaler.scale_[-1])
        y_test = inverse_transform(y_test, scaler.mean_[-1], scaler.scale_[-1])

    # Calculate RMSE
    rmse_scaled = mean_squared_error(y_test, y_pred, squared=False)

    # Update the RMSE in the DataFrame
    results_df.loc[index, 'RMSE_global_test'] = rmse_scaled

# Save the updated results DataFrame to a new CSV file
new_results_file_path = os.path.join(combined_folder_path, "combine_ML_240724_1608_scaled_2.csv")
results_df.to_csv(new_results_file_path, index=False)
print(f"Updated results saved to {new_results_file_path}")