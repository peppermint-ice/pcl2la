import pickle
import xgboost as xgb
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from config import paths
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression


def plot_distributions(train_set, test_set, fold_num, scaler, before=True):
    train_rescaled = train_set.copy()
    test_rescaled = test_set.copy()

    # Apply inverse transform to the 'measured_leaf_area' column
    train_rescaled['measured_leaf_area'] = inverse_transform(train_set['measured_leaf_area'], scaler.mean_[-1],
                                                             scaler.scale_[-1])
    test_rescaled['measured_leaf_area'] = inverse_transform(test_set['measured_leaf_area'], scaler.mean_[-1],
                                                            scaler.scale_[-1])

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.kdeplot(train_rescaled['measured_leaf_area'], label='Train', fill=True)
    sns.kdeplot(test_rescaled['measured_leaf_area'], label='Test', fill=True)
    title = 'Before Elimination' if before else 'After Elimination'
    plt.title(f'Distribution of Measured Leaf Area Fold {fold_num} ({title})')
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.histplot(train_rescaled['measured_leaf_area'], label='Train', color='blue', kde=False, alpha=0.5)
    sns.histplot(test_rescaled['measured_leaf_area'], label='Test', color='red', kde=False, alpha=0.5)
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


def plot_aggregated_predictions(y_test_all, y_test_pred_all, r2_aggregated, min_limit=None, max_limit=None):
    if min_limit is not None or max_limit is not None:
        y_test_all, y_test_pred_all = apply_limits(y_test_all, y_test_pred_all, min_limit, max_limit)
        r2_aggregated = r2_score(y_test_all, y_test_pred_all)

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_pred_all, y_test_all, alpha=0.5, label='Predicted vs Measured')
    plt.plot([min(y_test_all), max(y_test_all)], [min(y_test_all), max(y_test_all)], 'k--', lw=2)
    plt.ylabel('Measured')
    plt.xlabel('Predicted')
    plt.title(f'Aggregated Predictions vs True Values\nR2: {r2_aggregated:.4f}')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


def plot_aggregated_distributions(y_train_all, y_test_all, min_limit=None, max_limit=None):
    if min_limit is not None or max_limit is not None:
        y_train_all = [y for y in y_train_all if
                       (min_limit is None or y >= min_limit) and (max_limit is None or y <= max_limit)]
        y_test_all = [y for y in y_test_all if
                      (min_limit is None or y >= min_limit) and (max_limit is None or y <= max_limit)]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.kdeplot(y_train_all, label='Train', fill=True)
    sns.kdeplot(y_test_all, label='Test', fill=True)
    plt.title(f'Distribution of Aggregated Measured Leaf Area')
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.histplot(y_train_all, label='Train', color='blue', kde=False, alpha=0.5)
    sns.histplot(y_test_all, label='Test', color='red', kde=False, alpha=0.5)
    plt.title(f'Histogram of Aggregated Measured Leaf Area')
    plt.legend()

    plt.tight_layout()
    plt.show()


def inverse_transform(scaled_values, mean, scale):
    return scaled_values * scale + mean


def apply_limits(y_true, y_pred, min_limit, max_limit):
    filtered_y_true = []
    filtered_y_pred = []
    for true, pred in zip(y_true, y_pred):
        if (min_limit is None or true >= min_limit) and (max_limit is None or true <= max_limit):
            filtered_y_true.append(true)
            filtered_y_pred.append(pred)
    return filtered_y_true, filtered_y_pred


if __name__ == '__main__':
    # Define parameters
    algorithm_name = "marching_cubes"
    parameter_value = "0.8"
    assessment_name = "o3d"
    dataset_type = "simple"
    elimination_status = "elim"
    regression_model = "rf"
    num_folds = 4
    min_limit = 0  # Optional minimum limit
    max_limit = 5400  # Optional maximum limit

    folder_paths = paths.get_paths()
    models_folder_path = folder_paths["models"]
    train_folder_path = folder_paths["train_sets"]
    test_folder_path = folder_paths["test_sets"]
    ready_for_training_path = folder_paths["ready_for_training"]

    # Generate base file name
    base_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_{elimination_status}_2_4"

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

    # Load the unscaled dataset and fit the scaler
    unscaled_data_path = os.path.join(ready_for_training_path, "marching_cubes_0.8_o3d_simple_noElim.csv")
    unscaled_data = pd.read_csv(unscaled_data_path)
    scaler = StandardScaler()
    scaler.fit(unscaled_data)

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

        # Rescale predictions and true values to original scale
        y_train_rescaled = inverse_transform(y_train, scaler.mean_[-1], scaler.scale_[-1])
        y_train_pred_rescaled = inverse_transform(y_train_pred, scaler.mean_[-1], scaler.scale_[-1])
        y_test_rescaled = inverse_transform(y_test, scaler.mean_[-1], scaler.scale_[-1])
        y_test_pred_rescaled = inverse_transform(y_test_pred, scaler.mean_[-1], scaler.scale_[-1])

        # Aggregate all predictions and true values
        y_train_all.extend(y_train_rescaled)
        y_train_pred_all.extend(y_train_pred_rescaled)
        y_test_all.extend(y_test_rescaled)
        y_test_pred_all.extend(y_test_pred_rescaled)
        y_test_all_scaled.extend(y_test)

    # Apply limits to filter out points outside the desired range
    if min_limit is not None or max_limit is not None:
        y_test_all, y_test_pred_all = apply_limits(y_test_all, y_test_pred_all, min_limit, max_limit)
        r2_aggregated = r2_score(y_test_all, y_test_pred_all)
    else:
        r2_aggregated = r2_score(y_test_all, y_test_pred_all)

    # Save the aggregated predictions and corresponding measured values to a CSV file
    df_results = pd.DataFrame({
        'Measured': y_test_all,
        'Predicted': y_test_pred_all
    })
    df_results.to_csv('values.csv', index=False)

    # Plot distributions for the aggregated data
    plot_aggregated_distributions(y_train_all, y_test_all, min_limit, max_limit)

    # Plot aggregated predictions against true values
    plot_aggregated_predictions(y_test_all, y_test_pred_all, r2_aggregated, min_limit, max_limit)
