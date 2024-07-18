import pickle
import xgboost as xgb
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import seaborn as sns
from config import paths
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression


def plot_distributions(train_set, test_set, fold_num, before=True):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.kdeplot(train_set['measured_leaf_area'], label='Train', fill=True)
    sns.kdeplot(test_set['measured_leaf_area'], label='Test', fill=True)
    title = 'Before Elimination' if before else 'After Elimination'
    plt.title(f'Distribution of Measured Leaf Area Fold {fold_num} ({title})')
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.histplot(train_set['measured_leaf_area'], label='Train', color='blue', kde=False, alpha=0.5)
    sns.histplot(test_set['measured_leaf_area'], label='Test', color='red', kde=False, alpha=0.5)
    plt.title(f'Histogram of Measured Leaf Area Fold {fold_num} ({title})')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_predictions(y_train, y_train_pred, y_test, y_test_pred, r2_train, r2_test, fold_num, before=True):
    title = 'Before Elimination' if before else 'After Elimination'

    # Plot predictions vs actual values for train set
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(y_train_pred, y_train, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
    plt.ylabel('Measured')
    plt.xlabel('Predicted')
    plt.title(f'Train Set Predictions Fold {fold_num} ({title})\nR2: {r2_train:.4f}')

    # Plot predictions vs actual values for test set
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_pred, y_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.ylabel('Measured')
    plt.xlabel('Predicted')
    plt.title(f'Test Set Predictions Fold {fold_num} ({title})\nR2: {r2_test:.4f}')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Define parameters
    algorithm_name = "marching_cubes"
    parameter_value = "0.8"  # Updated parameter value
    assessment_name = "o3d"
    dataset_type = "simple"
    elimination_status = "elim"
    regression_model = "rf"
    num_folds = 4
    apply_limit = False  # Set to False if you don't want to apply the limit
    limit_value = 8000  # Updated limit value
    cutoff = 95

    folder_paths = paths.get_paths()
    models_folder_path = folder_paths["models"]
    train_folder_path = folder_paths["train_sets"]
    test_folder_path = folder_paths["test_sets"]
    ready_for_training_path = folder_paths["ready_for_training"]
    ready_for_training_path_2 = folder_paths["ready_for_training_2"]

    # Generate base file name
    base_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_{elimination_status}"

    file_names = [f"{base_file_name}_model_kf_{regression_model}_fold_{i}" for i in range(num_folds)]

    models = []
    train_sets = []
    test_sets = []

    for i in range(num_folds):
        file_name = file_names[i]

        # Load model
        model_path = os.path.join(models_folder_path, file_name) + ".pkl"
        with open(model_path, 'rb') as file:
            old_model = pickle.load(file)
            if regression_model == "xgb":
                new_model_path = model_path.replace(".pkl", ".json")
                old_model.save_model(new_model_path)
                model = xgb.Booster()
                model.load_model(new_model_path)
            else:
                model = old_model
            models.append(model)

        # Load train and test sets
        train_file_name = f"{base_file_name}_train_kf_{regression_model}_fold_{i}.csv"
        test_file_name = f"{base_file_name}_test_kf_{regression_model}_fold_{i}.csv"

        train_file_path = os.path.join(train_folder_path, train_file_name)
        test_file_path = os.path.join(test_folder_path, test_file_name)

        train_set = pd.read_csv(train_file_path)
        test_set = pd.read_csv(test_file_path)

        train_sets.append(train_set)
        test_sets.append(test_set)

    y_train_all = []
    y_train_pred_all = []
    y_test_all = []
    y_test_pred_all = []
    y_test_all_scaled = []

    for i in range(num_folds):
        model = models[i]
        train_set = train_sets[i]
        test_set = test_sets[i]

        X_train = train_set.drop('measured_leaf_area', axis=1)
        y_train = train_set['measured_leaf_area']
        X_test = test_set.drop('measured_leaf_area', axis=1)
        y_test = test_set['measured_leaf_area']

        if regression_model == "xgb":
            dtrain = xgb.DMatrix(X_train)
            dtest = xgb.DMatrix(X_test)
            y_train_pred = model.predict(dtrain)
            y_test_pred = model.predict(dtest)
        else:
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

        # Aggregate all predictions and true values
        y_train_all.extend(y_train)
        y_train_pred_all.extend(y_train_pred)
        y_test_all.extend(y_test)
        y_test_pred_all.extend(y_test_pred)
        y_test_all_scaled.extend(y_test)

    # Calculate R2 score for aggregated predictions
    r2_aggregated = r2_score(y_test_all, y_test_pred_all)

    if apply_limit:
        # Apply top limit for measured leaf area
        y_test_all_limited = [y for y in y_test_all if y <= limit_value]
        y_test_all_scaled_limited = [y_test_all_scaled[i] for i in range(len(y_test_all)) if y_test_all[i] <= limit_value]
        aggregated_y_test_pred_limited = [y_test_pred_all[i] for i in range(len(y_test_all)) if y_test_all[i] <= limit_value]

        # Calculate R2 score with the limitation
        r2_limited = r2_score(y_test_all_limited, aggregated_y_test_pred_limited)

        # Identify the 10% worst points for aggregated predictions with the limitation
        errors_limited = np.abs(np.array(y_test_all_limited) - np.array(aggregated_y_test_pred_limited))
        threshold_limited = np.percentile(errors_limited, cutoff)
        worst_indices_limited = np.where(errors_limited >= threshold_limited)[0]
        worst_measured_leaf_areas_limited = np.array(y_test_all_scaled_limited)[worst_indices_limited]

        # Remove the worst 10% points from the dataset
        unscaled_data_path = os.path.join(ready_for_training_path, "marching_cubes_0.8_o3d_simple_elim.csv")
        unscaled_data = pd.read_csv(unscaled_data_path)
        cleaned_data = unscaled_data.drop(index=unscaled_data.index[worst_indices_limited])
        cleaned_data_path = os.path.join(ready_for_training_path_2, "marching_cubes_0.8_o3d_simple_elim_3.csv")
        os.makedirs(ready_for_training_path_2, exist_ok=True)
        cleaned_data.to_csv(cleaned_data_path, index=False)

        # Plot aggregated predictions against true values with limitation and highlight the worst points
        plt.figure(figsize=(8, 6))
        plt.scatter(aggregated_y_test_pred_limited, y_test_all_limited, alpha=0.5, label='Predicted vs Measured')
        plt.scatter(np.array(aggregated_y_test_pred_limited)[worst_indices_limited], np.array(y_test_all_limited)[worst_indices_limited], color='red', alpha=0.5, label='10% Worst Points')
        plt.plot([min(y_test_all_limited), max(y_test_all_limited)], [min(aggregated_y_test_pred_limited), max(aggregated_y_test_pred_limited)], 'k--', lw=2)
        plt.ylabel('Measured')
        plt.xlabel('Predicted')
        plt.title(f'Aggregated Predictions vs True Values (Limited to {limit_value})\nR2: {r2_limited:.4f}')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Print the 10% worst points for aggregated predictions with limitation
        print(f"10% worst points (measured leaf area values) with limitation to {limit_value}:")
        print(worst_measured_leaf_areas_limited)
    else:
        # Identify the 10% worst points for aggregated predictions
        errors = np.abs(np.array(y_test_all) - np.array(y_test_pred_all))
        threshold = np.percentile(errors, cutoff)
        worst_indices = np.where(errors >= threshold)[0]
        worst_measured_leaf_areas = np.array(y_test_all_scaled)[worst_indices]

        # Remove the worst 10% points from the dataset
        unscaled_data_path = os.path.join(ready_for_training_path, "marching_cubes_0.8_o3d_simple_elim.csv")
        unscaled_data = pd.read_csv(unscaled_data_path)
        cleaned_data = unscaled_data.drop(index=unscaled_data.index[worst_indices])
        print(cleaned_data)
        cleaned_data_path = os.path.join(ready_for_training_path_2, "marching_cubes_0.8_o3d_simple_elim_3.csv")
        os.makedirs(ready_for_training_path_2, exist_ok=True)
        cleaned_data.to_csv(cleaned_data_path, index=False)

        # Plot aggregated predictions against true values and highlight the worst points
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_pred_all, y_test_all, alpha=0.5, label='Predicted vs Measured')
        plt.scatter(np.array(y_test_pred_all)[worst_indices], np.array(y_test_all)[worst_indices], color='red', alpha=0.5, label='10% Worst Points')
        plt.plot([min(y_test_all), max(y_test_all)], [min(y_test_pred_all), max(y_test_pred_all)], 'k--', lw=2)
        plt.ylabel('Measured')
        plt.xlabel('Predicted')
        plt.title(f'Aggregated Predictions vs True Values\nR2: {r2_aggregated:.4f}')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Print the 10% worst points for aggregated predictions
        print("10% worst points (measured leaf area values):")
        print(worst_measured_leaf_areas)
