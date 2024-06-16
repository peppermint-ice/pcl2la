from config import paths
import pickle
import xgboost as xgb
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np

if __name__ == '__main__':
    folder_paths = paths.get_paths()
    models_folder_path = folder_paths["models"]
    train_folder_path = folder_paths["train_sets"]
    test_folder_path = folder_paths["test_sets"]

    base_file_name = "marching_cubes_0.9_o3d_simple_noElim"

    file_names = []
    for i in range(5):
        file_names.append(f"{base_file_name}_model_kf_xgb_fold_{i}")

    models = []
    train_sets = []
    test_sets = []

    for i in range(5):
        file_name = file_names[i]

        # Load model
        model_path = os.path.join(models_folder_path, file_name) + ".pkl"
        with open(model_path, 'rb') as file:
            old_model = pickle.load(file)
            new_model_path = model_path.replace(".pkl", ".json")
            old_model.save_model(new_model_path)
            model = xgb.Booster()
            model.load_model(new_model_path)
            models.append(model)

        # Load train and test sets
        train_file_name = f"{base_file_name}_train_kf_xgb_fold_{i}.csv"
        test_file_name = f"{base_file_name}_test_kf_xgb_fold_{i}.csv"

        train_file_path = os.path.join(train_folder_path, train_file_name)
        test_file_path = os.path.join(test_folder_path, test_file_name)

        train_set = pd.read_csv(train_file_path)
        test_set = pd.read_csv(test_file_path)

        train_sets.append(train_set)
        test_sets.append(test_set)

    r2_train_scores = []
    r2_test_scores = []

    for i in range(5):
        model = models[i]
        train_set = train_sets[i]
        test_set = test_sets[i]

        X_train = train_set.drop('measured_leaf_area', axis=1)
        y_train = train_set['measured_leaf_area']
        X_test = test_set.drop('measured_leaf_area', axis=1)
        y_test = test_set['measured_leaf_area']

        # Make predictions
        dtrain = xgb.DMatrix(X_train)
        dtest = xgb.DMatrix(X_test)
        y_train_pred = model.predict(dtrain)
        y_test_pred = model.predict(dtest)

        # Plot predictions vs actual values for train set
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(y_train, y_train_pred, alpha=0.5)
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
        plt.xlabel('Measured')
        plt.ylabel('Predicted')
        plt.title(f'Train Set Predictions Fold {i + 1}')

        # Plot predictions vs actual values for test set
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_test_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('Measured')
        plt.ylabel('Predicted')
        plt.title(f'Test Set Predictions Fold {i + 1}')

        plt.tight_layout()
        plt.show()

        # Calculate R2 scores
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        r2_train_scores.append(r2_train)
        r2_test_scores.append(r2_test)

    # Calculate mean and median R2 scores
    mean_r2_train = np.mean(r2_train_scores)
    median_r2_train = np.median(r2_train_scores)
    mean_r2_test = np.mean(r2_test_scores)
    median_r2_test = np.median(r2_test_scores)

    print(f"Mean R2 score for Train sets: {mean_r2_train}")
    print(f"Median R2 score for Train sets: {median_r2_train}")
    print(f"Mean R2 score for Test sets: {mean_r2_test}")
    print(f"Median R2 score for Test sets: {median_r2_test}")
