import pickle
import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             mean_absolute_percentage_error, median_absolute_error, explained_variance_score)
from config import paths

# Define the function to unscale predictions and datasets
def unscale(df_scaled, df_original, scaler, type: str):
    original_columns = df_original.columns.difference(["experiment_number"])
    experiment_number = df_original["experiment_number"]
    for column in original_columns:
        if column not in df_scaled.columns:
            df_scaled[column] = 0
    df_filled = df_scaled[original_columns]
    df_unscaled = pd.DataFrame(scaler.inverse_transform(df_filled), columns=df_filled.columns)
    if type == 'original':
        df_unscaled = pd.concat([df_unscaled, experiment_number], axis=1)
    elif type == 'testset':
        df_unscaled = df_unscaled
    elif type == 'prediction':
        df_unscaled = df_unscaled['measured_leaf_area']
    return df_unscaled

# Load paths and data
folder_paths = paths.get_paths()
combined_folder_path = folder_paths['combined']
models_folder_path = folder_paths["models"]
global_test_sets_path = folder_paths["global_test_sets"]
ready_for_training_folder_path = folder_paths["ready_for_training"]
scalers_folder_path = folder_paths["scalers"]

r2_df = pd.read_csv(os.path.join(combined_folder_path, 'combine_ML_060824_1807_filtered_r2.csv'))

# Get the best models based on R2, including byyear grouping
def get_best_values(df, value_col, agg_func):
    return df.loc[df.groupby(['algorithm_name', 'regression_model', 'byyear'])[value_col].transform(agg_func) == df[value_col]]

best_r2 = get_best_values(r2_df, 'R2_global_test', 'max')

# Initialize results DataFrame
results = []

# Loop through the best models and calculate metrics
for _, row in best_r2.iterrows():
    # Extract model details
    algorithm_name = row['algorithm_name']
    parameter_value = row['parameter_value']
    regression_model = row['regression_model']
    byyear = row['byyear']
    assessment_name = row['assessment_name']
    dataset_type = row['dataset_type']
    elimination_status = row['elimination_status']
    print(algorithm_name, parameter_value, assessment_name, dataset_type, elimination_status, byyear, regression_model)

    # Ensure parameter_value is formatted correctly
    if isinstance(parameter_value, float) and parameter_value.is_integer():
        parameter_value = int(parameter_value)
    parameter_value = str(parameter_value)

    # Construct file paths dynamically
    global_test_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_{elimination_status}_{regression_model}_global_test_set_{byyear}.csv"
    scaler_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_scaler.pkl"
    original_dataset_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_noElim.csv"

    global_test_file_path = os.path.join(global_test_sets_path, global_test_file_name)
    scaler_file_path = os.path.join(scalers_folder_path, scaler_file_name)
    original_dataset_file_path = os.path.join(ready_for_training_folder_path, original_dataset_file_name)

    # Determine model file extension
    if regression_model == "xgb":
        file_type = "json"
    else:
        file_type = "pkl"

    model_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_{elimination_status}_best_model_{regression_model}_{byyear}.{file_type}"
    model_file_path = os.path.join(models_folder_path, model_file_name)

    # Load the model
    if regression_model == 'xgb':
        model = xgb.Booster()
        model.load_model(model_file_path)
    else:
        with open(model_file_path, 'rb') as file:
            model = pickle.load(file)

    # Load datasets
    global_test_df = pd.read_csv(global_test_file_path)
    df_original = pd.read_csv(original_dataset_file_path)

    # Extract features and target
    X_test = global_test_df.drop(columns=['measured_leaf_area'])
    y_test = global_test_df['measured_leaf_area']

    if regression_model == 'xgb':
        X_test = xgb.DMatrix(X_test)

    # Make predictions
    y_pred = model.predict(X_test)

    # Unscale if elimination status is 'elim'
    if elimination_status == 'elim':
        # Load scaler
        with open(scaler_file_path, 'rb') as file:
            scaler = pickle.load(file)

        # Unscale test dataset
        global_test_df = unscale(global_test_df, df_original, scaler, "testset")

        # Unscale predictions
        y_pred = unscale(pd.DataFrame({"measured_leaf_area": y_pred}), df_original, scaler, "prediction").values

        # Extract updated features and target
        X_test = global_test_df.drop(columns=['measured_leaf_area'])
        y_test = global_test_df['measured_leaf_area']

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    median_ae = median_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    bias = (y_pred - y_test).mean()

    # Append results
    results.append({
        'Algorithm': algorithm_name,
        'Regression Model': regression_model,
        'ByYear': byyear,
        'R2': r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'Median AE': median_ae,
        'EVS': evs,
        'Bias': bias
    })
    print(r2)

# Convert results to a DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv('best_model_metrics_byyear.csv', index=False)
print(results_df.to_string())
