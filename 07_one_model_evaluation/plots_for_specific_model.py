import pickle
import seaborn as sns
import xgboost as xgb
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             mean_absolute_percentage_error, median_absolute_error, explained_variance_score)
from sklearn.decomposition import PCA
from config import paths


def unscale(df_scaled, df_original, scaler, type: str):
    original_columns = df_original.columns.difference(["experiment_number"])
    experiment_number = df_original["experiment_number"]
    # Fill empty columns
    for column in original_columns:
        if column not in df_scaled.columns:
            df_scaled[column] = 0
    df_filled = df_scaled[original_columns]
    # Unscale it using scaler
    df_unscaled = pd.DataFrame(scaler.inverse_transform(df_filled), columns=df_filled.columns)
    if type == 'original':
        df_unscaled = pd.concat([df_unscaled, experiment_number], axis=1)
    else:
        if type == 'testset':
            selected_features = df_scaled.columns
            df_unscaled = df_unscaled[selected_features]
        elif type == 'prediction':
            df_unscaled = df_unscaled['measured_leaf_area']
    return df_unscaled


# Function to format text
def format_text(text):
    return text.replace('_', ' ').title()


if __name__ == '__main__':
    # Filename
    algorithm_name = "marching_cubes"
    parameter_value = "1.3"
    assessment_name = "trimesh"
    dataset_type = "repaired"
    elimination_status = "noElim"
    regression_model = "xgb"
    byyear = "noyear"
    if regression_model == "xgb":
        file_type = "json"
    else:
        file_type = "pkl"

    model_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_{elimination_status}_best_model_{regression_model}_{byyear}.{file_type}"
    print(algorithm_name, parameter_value, assessment_name, dataset_type, elimination_status, regression_model, file_type)

    # Set folder paths
    folder_paths = paths.get_paths()
    models_folder_path = folder_paths["models"]
    global_test_sets_path = folder_paths["global_test_sets"]
    train_folder_path = folder_paths["train_sets"]
    test_folder_path = folder_paths["test_sets"]
    ready_for_training_folder_path = folder_paths["ready_for_training"]
    scalers_folder_path = folder_paths["scalers"]

    # Load a model
    model_file_path = os.path.join(models_folder_path, model_file_name)
    if regression_model == 'xgb':
        model = xgb.Booster()
        model.load_model(model_file_path)
    else:
        with open(model_file_path, 'rb') as file:
            model = pickle.load(file)
    print("Model loaded")

    # Load test set and original dataset
    global_test_set_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_{elimination_status}_{regression_model}_global_test_set_{byyear}.csv"
    global_test_set_file_path = os.path.join(global_test_sets_path, global_test_set_file_name)
    global_test_df = pd.read_csv(global_test_set_file_path)
    print(len(global_test_df))

    original_dataset_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_noElim.csv"
    original_dataset_file_path = os.path.join(ready_for_training_folder_path, original_dataset_file_name)
    df_original = pd.read_csv(original_dataset_file_path)

    # Extract features and target
    X_test = global_test_df.drop(columns=['measured_leaf_area'])
    y_test = global_test_df['measured_leaf_area']

    if regression_model == 'xgb':
        # Convert the test set into DMatrix, which is the data structure used by XGBoost
        X_test = xgb.DMatrix(X_test)

    # Make predictions
    y_pred = model.predict(X_test)

    # Unscale the test df if the model has been scaled
    if elimination_status == 'elim':
        # Load scaler
        scaler_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_scaler.pkl"
        scaler_file_path = os.path.join(scalers_folder_path, scaler_file_name)
        scaler = pickle.load(open(scaler_file_path, 'rb'))
        # Unscale test dataset
        global_test_df = unscale(global_test_df, df_original, scaler, "testset")
        # Unscale predictions
        y_pred = unscale(pd.DataFrame({"measured_leaf_area": y_pred}), df_original, scaler, "prediction").values
        print(y_pred)
        # Extract features and target
        X_test = global_test_df.drop(columns=['measured_leaf_area'])
        y_test = global_test_df['measured_leaf_area']
        print(y_test)

    # Calculate and print performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    median_ae = median_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R²): {r2}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")
    print(f"Median Absolute Error: {median_ae}")
    print(f"Explained Variance Score: {evs}")

    # Format algorithm_name and parameter_value for display
    formatted_algorithm_name = format_text(algorithm_name)
    formatted_parameter_value = format_text(parameter_value)

    parameter_value_type = {
        'marching_cubes': 'Threshold',
        'poisson': 'Depth',
        'alpha': 'Alpha',
        'ball_pivoting': 'Smallest Ball Radius'
    }
    model_names = {
        'linear': 'Linear',
        'en': 'Elastic Net',
        'lasso': 'Lasso',
        'ridge': 'Ridge',
        'rf': 'RandomForest',
        'xgb': 'XGBoost',
    }


    #   PLOTS
    # Plot actual vs. predicted values with a 1:1 line and metrics
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Measured Leaf Area, Cm²')
    plt.ylabel('Predicted Leaf Area, Cm²')
    plt.title('Actual Vs. Predicted Measured Leaf Area')
    plt.axis('square')
    plt.xlim([min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())])
    plt.ylim([min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())])

    # Fit a linear regression model between y_test and y_pred
    linear_regressor = LinearRegression()
    y_pred_for_lin = y_pred.reshape(-1, 1)  # Reshape y_pred for fitting the linear model
    linear_regressor.fit(y_pred_for_lin, y_test)

    # Make predictions using the linear regression model
    y_pred_linear = linear_regressor.predict(y_pred_for_lin)

    # Calculate R² for the linear regression model
    linfit_r2 = r2_score(y_test, y_pred_linear)

    # Plot actual vs predicted values with the linear fit line
    # plt.figure(figsize=(10, 10))
    # plt.scatter(y_test, y_pred, alpha=0.3)
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=2, color='blue')
    # plt.xlabel('Measured Leaf Area, Cm²')
    # plt.ylabel('Predicted Leaf Area, Cm²')
    # plt.title('Actual Vs. Predicted Measured Leaf Area')
    # plt.axis('square')
    # plt.xlim([min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())])
    # plt.ylim([min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())])
    # plt.text(0.05, 0.75, f'Linear Fit R²: {round(linfit_r2, 2)}', transform=plt.gca().transAxes,
    #          fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    if regression_model == 'linear':
        plt.text(0.05, 0.95, f'R² = {r2:.2f}', transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    else:
        plt.text(0.05, 0.95, f'R² = {r2:.2f}\nRMSE = {rmse:.2f}', transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    plt.text(0.05, 0.85, f'3D Reconstruction Algorithm: {formatted_algorithm_name}\n{parameter_value_type[algorithm_name]} Value: {formatted_parameter_value}\nRegression Model: {model_names[regression_model]}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    plt.show()

    # Plot feature importance
    importance = model.get_score(importance_type='weight')
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    features = [format_text(item[0]) for item in importance]
    scores = [item[1] for item in importance]

    plt.figure(figsize=(16, 6))
    plt.barh(features, scores)
    plt.xlabel('Importance Score')
    plt.title('Feature Importance')
    plt.show()

    # Plot histogram of measured leaf area
    # Unscale original df if necessary
    if elimination_status == 'elim':
        df_original = unscale(df_original, df_original, scaler, "original")
    plt.figure(figsize=(12, 6))
    sns.histplot(x=df_original['measured_leaf_area'], hue=df_original['experiment_number'], kde=True, palette='deep', alpha=0.7)
    mean_value = df_original['measured_leaf_area'].mean()
    median_value = df_original['measured_leaf_area'].median()
    plt.axvline(mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')
    plt.axvline(median_value, color='b', linestyle='-', label=f'Median: {median_value:.2f}')
    plt.xlabel('Measured Leaf Area, Cm²')
    plt.ylabel('Frequency')
    plt.title('Histogram of Measured Leaf Area by Experiment Number')
    plt.legend(labels=['Experiment #2', 'Experiment #1', f'Mean: {mean_value:.2f}', f'Median: {median_value:.2f}'])
    plt.show()
