import pandas as pd
import os
import re

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, KFold
import pickle

from config import paths

if __name__ == '__main__':
    # Define the specific file path
    folder_paths = paths.get_paths()
    file_path = os.path.join(folder_paths["ready_for_training_2"], "marching_cubes_0.8_o3d_simple_noElim_by_year.csv")

    csv_folder_path = folder_paths["results"]
    model_folder_path = folder_paths["models"]
    train_folder_path = folder_paths["train_sets"]
    test_folder_path = folder_paths["test_sets"]

    # Read the CSV file
    df = pd.read_csv(file_path)

    print(df)

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
        'RMSE_score_calibration',
        'RMSE_score_validation',
        'R2_score_calibration',
        'R2_score_validation',
        'Successful_reconstructions_test',
        'Successful_reconstructions_train']
    current_results = dict.fromkeys(keys)
    results_rf = pd.DataFrame()

    try:
        # Split the data into training (2023) and testing (2024) sets
        train_df = df[df['Year'] == 2023]
        test_df = df[df['Year'] == 2024]

        X_train = train_df.drop(columns=['measured_leaf_area', 'Year'])
        y_train = train_df['measured_leaf_area']
        X_test = test_df.drop(columns=['measured_leaf_area', 'Year'])
        y_test = test_df['measured_leaf_area']

        print('Starting grid search')
        # Define distributions for hyperparameters
        param_dist = {
            'n_estimators': randint(50, 200),  # Number of trees in the forest
            'max_depth': [None] + list(randint(3, 10).rvs(5)),  # Maximum depth of the trees
            'min_samples_split': randint(2, 20),  # Minimum number of samples required to split a node
            'min_samples_leaf': randint(1, 10)  # Minimum number of samples required at each leaf node
        }

        # Perform random search with cross-validation on training data
        random_search = RandomizedSearchCV(RandomForestRegressor(), param_distributions=param_dist, n_iter=100,
                                           cv=5, scoring='neg_mean_squared_error', random_state=42)
        random_search.fit(X_train, y_train)

        # Get the best hyperparameters found by random search
        best_params = random_search.best_params_
        print("Best Hyperparameters:", best_params)

        # Refactor model training to use the best hyperparameters
        model = RandomForestRegressor(**best_params)
        model.fit(X_train, y_train)

        pred_cal = model.predict(X_train)
        pred_val = model.predict(X_test)
        mse_cal = mean_squared_error(y_train, pred_cal)
        mse_val = mean_squared_error(y_test, pred_val)
        r2_cal = r2_score(y_train, pred_cal)
        r2_val = r2_score(y_test, pred_val)
        print(r2_val)

        current_results = {
            'Parameter_name': parameter_name,
            'Parameter_value': parameter_value,
            'Assessment_name': assessment_name,
            'Repaired': repaired,
            'Eliminated': eliminated,
            'Regression_model': 'Random_forest',
            'RMSE_score_calibration': mse_cal,
            'RMSE_score_validation': mse_val,
            'R2_score_calibration': r2_cal,
            'R2_score_validation': r2_val,
            'Successful_reconstructions_test': len(X_test),
            'Successful_reconstructions_train': len(X_train)
        }
        results_rf = pd.concat([results_rf, pd.DataFrame([current_results])], ignore_index=True)

        # Save the model using pickle
        model_filename = f"{parameter_name}_{parameter_value}_{assessment_name}_{repaired}_{eliminated}_by_model_rf.pkl"
        model_filepath = os.path.join(model_folder_path, model_filename)
        with open(model_filepath, 'wb') as f:
            pickle.dump(model, f)

        # Save train and test datasets as CSV files
        train_filename = f"{parameter_name}_{parameter_value}_{assessment_name}_{repaired}_{eliminated}_by_train_rf.csv"
        train_filepath = os.path.join(train_folder_path, train_filename)
        train_set = X_train.copy()
        train_set['measured_leaf_area'] = y_train
        train_set.to_csv(train_filepath, index=False)

        test_filename = f"{parameter_name}_{parameter_value}_{assessment_name}_{repaired}_{eliminated}_by_test_rf.csv"
        test_filepath = os.path.join(test_folder_path, test_filename)
        test_set = X_test.copy()
        test_set['measured_leaf_area'] = y_test
        test_set.to_csv(test_filepath, index=False)

        output_file = f"{parameter_name}_{parameter_value}_{assessment_name}_{repaired}_{eliminated}_by_results_rf.csv"
        output_file_path = os.path.join(csv_folder_path, output_file)
        results_rf.to_csv(output_file_path, index=False)
        print(results_rf.shape)
    except ValueError:
        print('A small dataset. Cannot calculate')
