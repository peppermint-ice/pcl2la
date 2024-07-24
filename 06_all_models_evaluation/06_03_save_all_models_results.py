import pickle
import xgboost as xgb
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from config import paths
from datetime import datetime

def run_any_model(regression_model, x, model):
    # Run the model. XGBoost requires specific approach
    if regression_model == 'xgb':
        dmatrix = xgb.DMatrix(x)
        y_pred = model.predict(dmatrix)
    else:
        y_pred = model.predict(x)
    return y_pred

def plot_prediction(y_pred, y, model_parameters, subset_name):
    r2 = r2_score(y, y_pred)
    if r2 > 0.85:
        plt.scatter(y, y_pred)
        plt.title(f"R2 score for {model_parameters[0]} {model_parameters[1]} {model_parameters[5]} {subset_name}: {round(r2, 2)}")
        plt.show()

def adjust_features(df, expected_features):
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
    return df[expected_features]

if __name__ == '__main__':
    # Set folder paths
    folder_paths = paths.get_paths()
    models_folder_path = folder_paths["models"]
    global_test_sets_path = folder_paths["global_test_sets"]
    train_folder_path = folder_paths["train_sets"]
    test_folder_path = folder_paths["test_sets"]
    scalers_folder_path = folder_paths["scalers"]
    ready_for_training_path = folder_paths["ready_for_training"]

    # Define the expected features (the ones used when the scaler was fit)
    expected_features = ['height', 'length', 'width', 'volume', 'surface_area', 'aspect_ratio', 'components_number', 'measured_leaf_area']

    # List to store results
    results = []

    # Run through each model file
    models = os.listdir(models_folder_path)
    for model in models:
        # Check that the file is actually a model
        if os.path.isfile(os.path.join(models_folder_path, model)) and os.path.splitext(model)[1] == ".pkl":
            # Set the pattern to use regex
            pattern = "(\w*_?\w*?)_(\w*[.]?\w*)_(\w*)_(\w*)_(\w*)_best_model_(\w*)_(\w*).(\w*)"
            match = re.search(pattern, model)
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
                # Save them into one list
                model_parameters = [algorithm_name, parameter_value, assessment_name, dataset_type, elimination_status, regression_model, byyear]
                print("File:")
                print(algorithm_name, parameter_value, assessment_name, dataset_type, elimination_status, regression_model, byyear)
                # Create file names
                test_file_names = []
                train_file_names = []
                for i in range(6):
                    test_file_names.append(f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_{elimination_status}_val_kf_{regression_model}_fold_{i}_{byyear}.csv")
                    train_file_names.append(f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_{elimination_status}_train_kf_{regression_model}_fold_{i}_{byyear}.csv")
                # Define global test set file name
                global_test_set_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_{elimination_status}_{regression_model}_global_test_set_{byyear}.csv"
                # Define files paths
                if regression_model == "xgb":
                    json_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_{elimination_status}_best_model_{regression_model}_{byyear}.json"
                    model_file_path = os.path.join(models_folder_path, json_file_name)
                else:
                    model_file_path = os.path.join(models_folder_path, model)
                global_test_set_file_path = os.path.join(global_test_sets_path, global_test_set_file_name)
                test_file_paths = [os.path.join(test_folder_path, test_file_names[i]) for i in range(len(test_file_names))]
                train_file_paths = [os.path.join(train_folder_path, train_file_names[i]) for i in range(len(train_file_names))]

                # Load model
                with open(model_file_path, 'rb') as file:
                    if regression_model == 'xgb':
                        model = xgb.Booster()
                        model.load_model(model_file_path)
                    else:
                        model = pickle.load(file)
                print("Model loaded")

                # Load test set
                global_test_df = pd.read_csv(global_test_set_file_path)
                # Load test and train datasets
                test_dfs = [pd.read_csv(test_file_paths[i]) for i in range(len(test_file_paths))]
                train_dfs = [pd.read_csv(train_file_paths[i]) for i in range(len(train_file_paths))]
                print("Files loaded")

                # Load scaler if elimination status is "elim"
                scaler = None
                if elimination_status == "elim":
                    scaler_file_path = os.path.join(scalers_folder_path, f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_scaler.pkl")
                    if os.path.exists(scaler_file_path):
                        with open(scaler_file_path, 'rb') as scaler_file:
                            scaler = pickle.load(scaler_file)
                        print("Scaler loaded")
                    else:
                        print(f"Scaler file not found: {scaler_file_path}")

                # GLOBAL TEST
                print("Global test")
                # Extract X and Y
                x = global_test_df.drop('measured_leaf_area', axis=1)
                y = global_test_df['measured_leaf_area']
                # Adjust features to match the scaler's expected features
                if elimination_status == "elim" and scaler is not None:
                    x = adjust_features(x, expected_features[:-1])
                    x = scaler.transform(x)
                # Run the model
                y_pred = run_any_model(regression_model, x, model)
                # Get R2 and RMSE
                r2_global_test = r2_score(y, y_pred)
                rmse_global_test = mean_squared_error(y, y_pred, squared=False)
                print("R2 score: ", r2_global_test)

                # TESTING SETS
                print("Testing sets")
                r2_testing = []
                for test_df in test_dfs:
                    x = test_df.drop('measured_leaf_area', axis=1)
                    y = test_df['measured_leaf_area']
                    # Adjust features to match the scaler's expected features
                    if elimination_status == "elim" and scaler is not None:
                        x = adjust_features(x, expected_features[:-1])
                        x = scaler.transform(x)
                    # Run the model
                    y_pred = run_any_model(regression_model, x, model)
                    # Get R2
                    r2 = r2_score(y, y_pred)
                    r2_testing.append(r2)
                    print("R2 score: ", r2)

                # TRAINING SETS
                print("Training sets")
                r2_training = []
                for train_df in train_dfs:
                    x = train_df.drop('measured_leaf_area', axis=1)
                    y = train_df['measured_leaf_area']
                    # Adjust features to match the scaler's expected features
                    if elimination_status == "elim" and scaler is not None:
                        x = adjust_features(x, expected_features[:-1])
                        x = scaler.transform(x)
                    # Run the model
                    y_pred = run_any_model(regression_model, x, model)
                    # Get R2
                    r2 = r2_score(y, y_pred)
                    r2_training.append(r2)
                    print("R2 score: ", r2)

                # Calculate means for training and testing
                mean_r2_training = np.mean(r2_training)
                mean_r2_testing = np.mean(r2_testing)

                # Save the results
                result = (model_parameters + r2_training + r2_testing +
                          [mean_r2_training, mean_r2_testing, r2_global_test, rmse_global_test])
                results.append(result)
                print("")

    # Save results to CSV
    current_time = datetime.now().strftime("%d%m%y_%H%M")
    columns = ([
                   "algorithm_name",
                   "parameter_value",
                   "assessment_name",
                   "dataset_type",
                   "elimination_status",
                   "regression_model",
                   "byyear"
               ] +
               [f"R2_training_{i}" for i in range(6)] +
               [f"R2_testing_{i}" for i in range(6)] +
               [
                   "mean_R2_training",
                   "mean_R2_testing",
                   "R2_global_test",
                   "RMSE_global_test"
               ])
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv(os.path.join(folder_paths["combined"], f"combine_ML_{current_time}.csv"), index=False)
    print("Results saved to CSV")