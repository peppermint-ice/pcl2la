import pandas as pd
import os
import sys
import re

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.preprocessing import KBinsDiscretizer
import pickle
from config import paths

if __name__ == '__main__':
    # Get the file path from the command-line argument
    file_path = sys.argv[1]

    folder_paths = paths.get_paths()
    kfold_results_path = folder_paths["kfold_results"]
    csv_folder_path = folder_paths["results"]
    model_folder_path = folder_paths["models"]
    train_folder_path = folder_paths["train_sets"]
    test_folder_path = folder_paths["test_sets"]
    global_test_path = folder_paths["global_test_sets"]

    # Read the CSV file specified by the command-line argument
    df = pd.read_csv(file_path)

    # Regular expression pattern
    pattern = r'^([a-zA-Z0-9_]+)_([0-9.]+)_([a-zA-Z0-9_]+)_([a-zA-Z0-9_]+)_(.*)\.csv$'
    file_name = os.path.basename(file_path)
    match = re.match(pattern, file_name)
    parameter_name = match.groups()[0]
    parameter_value = match.groups()[1]
    assessment_name = match.groups()[2]
    repaired = match.groups()[3]
    eliminated = match.groups()[4]

    keys = [
        'Parameter_name',
        'Parameter_value',
        'Assessment_name',
        'Repaired',
        'Eliminated',
        'Regression_model',
        'K_fold',
        'RMSE_score_calibration',
        'RMSE_score_validation',
        'R2_score_calibration',
        'R2_score_validation',
        'Successful_reconstructions_test',
        'Successful_reconstructions_train'
    ]
    current_results = dict.fromkeys(keys)
    results_rf = pd.DataFrame()

    try:
        print('starting grid search')
        # Split dataset into train and test using 'experiment_number' value. 2023 for exp 1, 2024 for exp 2
        train_df = df[df['experiment_number'] == 1]
        train_df.drop(columns=['experiment_number'], inplace=True)
        test_df = df[df['experiment_number'] == 2]
        test_df.drop(columns=['experiment_number'], inplace=True)


        # Save the global test set
        global_test_filename = f"{parameter_name}_{parameter_value}_{assessment_name}_{repaired}_{eliminated}_rf_global_test_set.csv"
        global_test_filepath = os.path.join(global_test_path, global_test_filename)
        test_df.to_csv(global_test_filepath, index=False)

        # Prepare training and test data
        X_train = train_df.drop(columns=['measured_leaf_area'])
        y_train = train_df['measured_leaf_area']
        X_test = test_df.drop(columns=['measured_leaf_area'])
        y_test = test_df['measured_leaf_area']

        # Define distributions for hyperparameters
        param_dist = {
            'n_estimators': randint(50, 200),  # Number of trees in the forest
            'max_depth': [None] + list(randint(3, 10).rvs(5)),  # Maximum depth of the trees
            'min_samples_split': randint(2, 20),  # Minimum number of samples required to split a node
            'min_samples_leaf': randint(1, 10)  # Minimum number of samples required at each leaf node
        }

        # Initialize KFold cross-validator
        num_splits = 6
        kf = KFold(n_splits=num_splits, shuffle=True)

        best_fold_index = -1
        best_r2_val = float('-inf')

        # Iterate through each fold
        for i, (train_index, val_index) in enumerate(kf.split(X_train)):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            # Perform random search with cross-validation
            random_search = RandomizedSearchCV(RandomForestRegressor(), param_distributions=param_dist, n_iter=100,
                                               cv=5,
                                               scoring='neg_mean_squared_error')
            random_search.fit(X_train_fold, y_train_fold)

            # Get the best hyperparameters found by random search
            best_params = random_search.best_params_
            print(f"Best Hyperparameters for fold {i}: {best_params}")

            # Train model with best hyperparameters
            model = RandomForestRegressor(**best_params)
            model.fit(X_train_fold, y_train_fold)

            pred_train = model.predict(X_train_fold)
            pred_val = model.predict(X_val_fold)
            mse_train = mean_squared_error(y_train_fold, pred_train)
            mse_val = mean_squared_error(y_val_fold, pred_val)
            r2_train = r2_score(y_train_fold, pred_train)
            r2_val = r2_score(y_val_fold, pred_val)

            current_results = {
                'Parameter_name': parameter_name,
                'Parameter_value': parameter_value,
                'Assessment_name': assessment_name,
                'Repaired': repaired,
                'Eliminated': eliminated,
                'Regression_model': 'Random_forest',
                'K_fold': i,
                'RMSE_score_calibration': mse_train,
                'RMSE_score_validation': mse_val,
                'R2_score_calibration': r2_train,
                'R2_score_validation': r2_val,
                'Successful_reconstructions_test': len(X_val_fold),
                'Successful_reconstructions_train': len(X_train_fold)
            }
            results_rf = pd.concat([results_rf, pd.DataFrame([current_results])], ignore_index=True)

            # Save train and validation datasets as CSV files
            train_filename = f"{parameter_name}_{parameter_value}_{assessment_name}_{repaired}_{eliminated}_train_kf_rf_fold_{i}.csv"
            train_filepath = os.path.join(train_folder_path, train_filename)
            train_set = X_train_fold.copy()
            train_set['measured_leaf_area'] = y_train_fold
            train_set.to_csv(train_filepath, index=False)

            val_filename = f"{parameter_name}_{parameter_value}_{assessment_name}_{repaired}_{eliminated}_val_kf_rf_fold_{i}.csv"
            val_filepath = os.path.join(test_folder_path, val_filename)
            val_set = X_val_fold.copy()
            val_set['measured_leaf_area'] = y_val_fold
            val_set.to_csv(val_filepath, index=False)

            # Check if this fold has the best validation R2
            if r2_val > best_r2_val:
                best_r2_val = r2_val
                best_fold_index = i
                best_model = model

        # Save the best model using pickle
        model_filename = f"{parameter_name}_{parameter_value}_{assessment_name}_{repaired}_{eliminated}_best_model_rf.pkl"
        model_filepath = os.path.join(model_folder_path, model_filename)
        with open(model_filepath, 'wb') as f:
            pickle.dump(best_model, f)

        # Evaluate the best model on the test set
        pred_test = best_model.predict(X_test)
        mse_test = mean_squared_error(y_test, pred_test)
        r2_test = r2_score(y_test, pred_test)

        # Write test set performance to CSV
        test_results = {
            'Parameter_name': parameter_name,
            'Parameter_value': parameter_value,
            'Assessment_name': assessment_name,
            'Repaired': repaired,
            'Eliminated': eliminated,
            'Regression_model': 'Random_forest',
            'RMSE_score_test': mse_test,
            'R2_score_test': r2_test
        }
        test_results_df = pd.DataFrame([test_results])
        test_output_file = f"{parameter_name}_{parameter_value}_{assessment_name}_{repaired}_{eliminated}_test_results_rf.csv"
        test_output_file_path = os.path.join(csv_folder_path, test_output_file)
        test_results_df.to_csv(test_output_file_path, index=False)

        # Save the k-fold results
        kfold_output_file = f"{parameter_name}_{parameter_value}_{assessment_name}_{repaired}_{eliminated}_kfold_results_rf.csv"
        kfold_output_file_path = os.path.join(kfold_results_path, kfold_output_file)
        results_rf.to_csv(kfold_output_file_path, index=False)

    except ValueError as e:
        print('An error occurred:', e)
