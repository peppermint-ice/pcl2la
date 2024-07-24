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


if __name__ == '__main__':
    # Set random seed for reproducibility
    # set_random_seed()

    # Set folder paths
    folder_paths = paths.get_paths()
    models_folder_path = folder_paths["models"]
    global_test_sets_path = folder_paths["global_test_sets"]
    train_folder_path = folder_paths["train_sets"]
    test_folder_path = folder_paths["test_sets"]
    ready_for_training_path = folder_paths["ready_for_training"]
    scalers_folder_path = folder_paths["scalers"]

    # List to store results
    results = []

    # Initiate dfs for R2s and worst points
    worst_points = pd.DataFrame()
    worst_points_current = pd.DataFrame()

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
                # We are only interested in noElim files
                if elimination_status == "noElim":
                    # Create file names
                    # Create test and train set lists
                    test_file_names = []
                    train_file_names = []
                    for i in range(6):
                        test_file_names.append(f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_{elimination_status}_val_kf_{regression_model}_fold_{i}.csv")
                        train_file_names.append(f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_{elimination_status}_train_kf_{regression_model}_fold_{i}.csv")
                    # Define global test set file name
                    global_test_set_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_{elimination_status}_{regression_model}_global_test_set.csv"
                    # Define files paths
                    # Model file path is different for rf and xgb because the XGBoost needs to load a json
                    if regression_model == "xgb":
                        json_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_{elimination_status}_best_model_{regression_model}.json"
                        model_file_path = os.path.join(models_folder_path, json_file_name)
                    else:
                        model_file_path = os.path.join(models_folder_path, model)
                    global_test_set_file_path = os.path.join(global_test_sets_path, global_test_set_file_name)
                    test_file_paths = [os.path.join(test_folder_path, test_file_names[i]) for i in range(len(test_file_names))]
                    train_file_paths = [os.path.join(train_folder_path, train_file_names[i]) for i in range(len(train_file_names))]

                    # Load files
                    # Load models
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
                    print(train_file_paths)

                    # # Get lengths of test and train sets
                    # test_set_length = len(global_test_df)
                    # for train_set in train_dfs:
                    #     all_data = pd.concat([train_set], ignore_index=True)
                    #
                    # train_set_length += len(train_set)

                    # GLOBAL TEST
                    print("Global test")
                    # Extract X and Y
                    x = global_test_df.drop('measured_leaf_area', axis=1)
                    y = global_test_df['measured_leaf_area']
                    # Run the model
                    y_pred = run_any_model(regression_model, x, model)
                    # Get R2
                    r2_global_test = r2_score(y, y_pred)
                    rmse_global_test = mean_squared_error(y, y_pred, squared=False)
                    print("R2 score: ", r2_global_test)
                    # plot_prediction(y, y_pred, model_parameters, 'GLOBAL test subset')

                    # TESTING SETS
                    print("Testing sets")
                    # A loop to do the procedure at once for each kfold
                    r2_testing = []
                    for test_df in test_dfs:
                        x = test_df.drop('measured_leaf_area', axis=1)
                        y = test_df['measured_leaf_area']
                        # Run the model
                        y_pred = run_any_model(regression_model, x, model)
                        # Get R2
                        r2 = r2_score(y, y_pred)
                        r2_testing.append(r2)
                        print("R2 score: ", r2)
                        # plot_prediction(y, y_pred, model_parameters, 'testing subset')

                    # TRAINING SETS
                    print("Training sets")
                    # A loop to do the procedure at once for each kfold
                    r2_training = []
                    for train_df in train_dfs:
                        x = train_df.drop('measured_leaf_area', axis=1)
                        y = train_df['measured_leaf_area']
                        # Run the model
                        y_pred = run_any_model(regression_model, x, model)
                        # Get R2
                        r2 = r2_score(y, y_pred)
                        r2_training.append(r2)
                        print("R2 score: ", r2)
                        # plot_prediction(y, y_pred, model_parameters, 'training subset')

                    # Calculate means for training and testing
                    mean_r2_training = np.mean(r2_training)
                    mean_r2_testing = np.mean(r2_testing)

                    # Save the results
                    # Append results to the list
                    result = (model_parameters + r2_training + r2_testing +
                              [mean_r2_training, mean_r2_testing, r2_global_test, rmse_global_test])
                    results.append(result)
                    print("")

    # Save results to CSV
    # Get current time and format it
    current_time = datetime.now().strftime("%d%m%y_%H%M")
    # Create a table
    columns = ([
                   "algorithm_name",
                   "parameter_value",
                   "assessment_name",
                   "dataset_type",
                   "elimination_status",
                   "regression_model"
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
