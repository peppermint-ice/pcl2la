import pandas as pd
import os
import sys

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, KFold

from config import paths



if __name__ == '__main__':
    # Get the file path from the command-line argument
    file_path = sys.argv[1]

    folder_paths = paths.get_paths()
    csv_folder_path = folder_paths["ml_results"]

    # Read the CSV file specified by the command-line argument
    df = pd.read_csv(file_path)
    df = df[df['Components_number'] != 0]
    parameter_value = df['parameter_value'].unique()[0]
    parameter_type = df['parameter_type'].unique()[0]
    print(df['parameter_type'].unique())

    keys = [
        'Parameter_name',
        'Parameter_value',
        'Regression_model',
        'Correlating_parameter',
        'RMSE_score_calibration',
        'RMSE_score_validation',
        'R2_score_calibration',
        'R2_score_validation',
        'Successful_reconstructions_test',
        'Successful_reconstructions_train']
    current_results = dict.fromkeys(keys)
    results_xgb = pd.DataFrame()

    try:
        print('starting grid search')
        # Define distributions for hyperparameters
        param_dist = {
            'n_estimators': randint(50, 200),  # Number of boosting rounds
            'max_depth': randint(3, 10),  # Maximum depth of the trees
            'learning_rate': uniform(0.01, 0.3),  # Learning rate
            'min_child_weight': randint(1, 6)  # Minimum sum of instance weight needed in a child
        }

        # Initialize KFold cross-validator
        num_splits = 5
        kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

        # Iterate through each fold
        for i, (train_index, test_index) in enumerate(kf.split(df)):
            X_train, X_test = (df.drop(
                columns=['Measured_leaf_area', 'File_name', 'parameter_type', 'parameter_value', 'Year', 'Elongation',
                         'Flatness', 'Sphericity', 'Compactness'],
                inplace=False, axis=1).iloc[train_index],
                               df.drop(columns=['Measured_leaf_area', 'File_name', 'parameter_type', 'parameter_value',
                                                'Year', 'Elongation', 'Flatness', 'Sphericity', 'Compactness'],
                                       inplace=False, axis=1).iloc[test_index])
            y_train, y_test = df['Measured_leaf_area'].iloc[train_index], df['Measured_leaf_area'].iloc[test_index]

            # Perform random search with cross-validation
            random_search = RandomizedSearchCV(XGBRegressor(), param_distributions=param_dist, n_iter=100, cv=5,
                                               scoring='neg_mean_squared_error')
            random_search.fit(X_train, y_train)

            # Get the best hyperparameters found by random search
            best_params = random_search.best_params_
            print("Best Hyperparameters:", best_params)

            # Refactor model training to use the best hyperparameters
            model = XGBRegressor(**best_params)
            model.fit(X_train, y_train)

            pred_cal = model.predict(X_train)
            pred_val = model.predict(X_test)
            mse_cal = mean_squared_error(y_train, pred_cal)
            mse_val = mean_squared_error(y_test, pred_val)
            r2_cal = r2_score(y_train, pred_cal)
            r2_val = r2_score(y_test, pred_val)

            current_results = {
                'Parameter_name': df['parameter_type'].unique()[0],
                'Parameter_value': df['parameter_value'].unique()[0],
                'Regression_model': 'XGBoost',
                'K_fold': i,
                'RMSE_score_calibration': mse_cal,
                'RMSE_score_validation': mse_val,
                'R2_score_calibration': r2_cal,
                'R2_score_validation': r2_val,
                'Successful_reconstructions_test': len(X_test),
                'Successful_reconstructions_train': len(X_train)
            }
            results_xgb = pd.concat([results_xgb, pd.DataFrame([current_results])], ignore_index=True)
            output_file = str(parameter_value) + parameter_type + '_results_kf_xgb.csv'
            output_file_path = os.path.join(csv_folder_path, output_file)
    except ValueError:
        print('A small dataset. Cannot calculate')
    results_xgb.to_csv(output_file_path, index=False)
