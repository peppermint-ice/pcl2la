import pandas as pd
import os
import pickle
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from config import paths


# Define function to reverse scaling
def inverse_transform(scaled_values, mean, scale):
    return scaled_values * scale + mean


# Set paths
folder_paths = paths.get_paths()
combined_folder_path = folder_paths["combined"]
results_file_path = os.path.join(combined_folder_path, "combine_ML_240724_1608.csv")
ready_for_training_path = folder_paths["ready_for_training"]
models_folder_path = folder_paths["models"]
global_test_sets_path = folder_paths["global_test_sets"]

# Load the results CSV file
df = pd.read_csv(results_file_path)

# Iterate through each row and reverse scale if elimination_status is 'elim'
for idx, row in df.iterrows():
    if row['elimination_status'] == 'elim':
        # Determine the correct model file extension
        if row['regression_model'] == 'xgb':
            model_file_ext = 'json'
        else:
            model_file_ext = 'pkl'

        # Load the model
        model_file_name = f"{row['algorithm_name']}_{row['parameter_value']}_{row['assessment_name']}_{row['dataset_type']}_{row['elimination_status']}_best_model_{row['regression_model']}_{row['byyear']}.{model_file_ext}"
        model_file_path = os.path.join(models_folder_path, model_file_name)

        if row['regression_model'] == 'xgb':
            model = xgb.Booster()
            model.load_model(model_file_path)
        else:
            with open(model_file_path, 'rb') as file:
                model = pickle.load(file)

        # Load the global test set
        global_test_set_file_name = f"{row['algorithm_name']}_{row['parameter_value']}_{row['assessment_name']}_{row['dataset_type']}_{row['elimination_status']}_{row['regression_model']}_global_test_set_{row['byyear']}.csv"
        global_test_set_file_path = os.path.join(global_test_sets_path, global_test_set_file_name)
        global_test_df = pd.read_csv(global_test_set_file_path)

        # Extract features and target
        X_test = global_test_df.drop(columns=['measured_leaf_area'])
        y_test = global_test_df['measured_leaf_area']

        # If elimination_status is 'elim', load the original unscaled dataset to get scaling parameters
        unscaled_data_path = os.path.join(ready_for_training_path,
                                          f"{row['algorithm_name']}_{row['parameter_value']}_{row['assessment_name']}_{row['dataset_type']}_noElim.csv")
        unscaled_data = pd.read_csv(unscaled_data_path)
        scaler = StandardScaler()
        scaler.fit(unscaled_data.drop(columns=['measured_leaf_area', 'experiment_number']))
        mean = scaler.mean_
        scale = scaler.scale_

        # Make predictions
        if row['regression_model'] == 'xgb':
            dtest = xgb.DMatrix(X_test)
            y_pred = model.predict(dtest)
        else:
            y_pred = model.predict(X_test)

        # Reverse scaling for y_true and y_pred
        y_test_unscaled = inverse_transform(y_test, mean[-1], scale[-1])
        y_pred_unscaled = inverse_transform(y_pred, mean[-1], scale[-1])

        # Recalculate RMSE
        rmse_scaled = mean_squared_error(y_test_unscaled, y_pred_unscaled, squared=False)

        # Update the DataFrame with new RMSE
        df.at[idx, 'RMSE_global_test'] = rmse_scaled

# Save the updated DataFrame to a new CSV file
output_file_path = os.path.join(combined_folder_path, "combine_ML_240724_1608_scaled.csv")
df.to_csv(output_file_path, index=False)

print(f"Updated results saved to {output_file_path}")
