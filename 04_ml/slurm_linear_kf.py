import pandas as pd
import os
import sys
import re


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from config import paths
import pickle

if __name__ == '__main__':
    # Get the file path from the command-line argument
    file_path = sys.argv[1]

    folder_paths = paths.get_paths()
    csv_folder_path = folder_paths["results"]
    model_folder_path = folder_paths["models"]
    train_folder_path = folder_paths["train_sets"]
    test_folder_path = folder_paths["test_sets"]

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
        'Repaired'
        'Eliminated',
        'Regression_model',
        'RMSE_score_calibration',
        'RMSE_score_validation',
        'R2_score_calibration',
        'R2_score_validation',
        'Successful_reconstructions_test',
        'Successful_reconstructions_train']
    current_results = dict.fromkeys(keys)
    results_linreg = pd.DataFrame()

    try:
        print('starting linear regression')
        # Initialize KFold cross-validator
        num_splits = 5
        kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

        # Iterate through each fold
        for i, (train_index, test_index) in enumerate(kf.split(df)):
            X_train, X_test = df.drop(columns=['measured_leaf_area'], inplace=False).iloc[train_index], df.drop(
                columns=['measured_leaf_area'], inplace=False).iloc[test_index]
            y_train, y_test = df['measured_leaf_area'].iloc[train_index], df['measured_leaf_area'].iloc[test_index]

            # Fit linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            pred_cal = model.predict(X_train)
            pred_val = model.predict(X_test)

            # Calculate metrics
            mse_cal = mean_squared_error(y_train, pred_cal)
            mse_val = mean_squared_error(y_test, pred_val)
            r2_cal = r2_score(y_train, pred_cal)
            r2_val = r2_score(y_test, pred_val)

            # Store results
            current_results = {
                'Parameter_name': parameter_name,
                'Parameter_value': parameter_value,
                'Assessment_name': assessment_name,
                'Repaired': repaired,
                'Eliminated': eliminated,
                'Regression_model': 'Linear_regression',
                'K_fold': i,
                'RMSE_score_calibration': mse_cal,
                'RMSE_score_validation': mse_val,
                'R2_score_calibration': r2_cal,
                'R2_score_validation': r2_val,
                'Successful_reconstructions_test': len(X_test),
                'Successful_reconstructions_train': len(X_train)
            }
            results_linreg = pd.concat([results_linreg, pd.DataFrame([current_results])], ignore_index=True)

            # Save the model using pickle
            model_filename = parameter_name + "_" + str(
                parameter_value) + "_" + assessment_name + "_" + repaired + "_" + eliminated + "_model_kf_lr_fold_" + str(
                i) + ".pkl"
            model_filepath = os.path.join(model_folder_path, model_filename)
            with open(model_filepath, 'wb') as f:
                pickle.dump(model, f)

            # Save train and test datasets as CSV files
            train_filename = parameter_name + "_" + str(
                parameter_value) + "_" + assessment_name + "_" + repaired + "_" + eliminated + "_train_kf_lr_fold_" + str(
                i) + ".csv"
            train_filepath = os.path.join(train_folder_path, train_filename)
            X_train.to_csv(train_filepath, index=False)

            test_filename = parameter_name + "_" + str(
                parameter_value) + "_" + assessment_name + "_" + repaired + "_" + eliminated + "_test_kf_lr_fold_" + str(
                i) + ".csv"
            test_filepath = os.path.join(test_folder_path, test_filename)
            X_test.to_csv(test_filepath, index=False)
            # Define output file path
            output_file = parameter_name + "_" + str(
                parameter_value) + "_" + assessment_name + "_" + repaired + "_" + eliminated + '_results_kf_lr' + '.csv'
            output_file_path = os.path.join(csv_folder_path, output_file)

    except ValueError:
        print('A small dataset. Cannot calculate')

    # Save results to CSV
    results_linreg.to_csv(output_file_path, index=False)