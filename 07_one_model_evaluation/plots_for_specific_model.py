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
    # Set folder paths
    folder_paths = paths.get_paths()
    models_folder_path = folder_paths["models"]
    global_test_sets_path = folder_paths["global_test_sets"]
    train_folder_path = folder_paths["train_sets"]
    test_folder_path = folder_paths["test_sets"]
    ready_for_training_folder_path = folder_paths["ready_for_training"]
    scalers_folder_path = folder_paths["scalers"]

    # Create filename using instructions
    algorithm_name = "alpha"
    parameter_value = "3"
    assessment_name = "o3d"
    dataset_type = "simple"
    elimination_status = "elim"
    regression_model = "xgb"
    byyear = "noyear"

    # Set plot options
    limit = False
    metric_type = False

    # Format algorithm_name and parameter_value for display
    formatted_algorithm_name = format_text(algorithm_name)
    formatted_parameter_value = format_text(parameter_value)

    # Dictionaries for names
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
        'mlp': 'MLP'
    }

    # Determine model file extension
    if regression_model == "xgb":
        file_type = "json"
    else:
        file_type = "pkl"

    # Compose model file name
    model_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_{elimination_status}_best_model_{regression_model}_{byyear}.{file_type}"
    print(algorithm_name, parameter_value, assessment_name, dataset_type, elimination_status, regression_model, file_type)

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
    print(f'Global test length: {len(global_test_df)}')

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
        # Extract features and target
        X_test = global_test_df.drop(columns=['measured_leaf_area'])
        y_test = global_test_df['measured_leaf_area']

    if limit:
        # Add limit
        limit = 5000
        mask = (y_test < limit) & (y_pred < limit)
        y_test = y_test[mask]
        y_pred = y_pred[mask]


    # Calculate mean and median
    mean_value = df_original['measured_leaf_area'].mean()
    median_value = df_original['measured_leaf_area'].median()

    # Calculate and print performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    median_ae = median_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    bias = (y_pred - y_test).mean()

    print(f"Mean: {mean_value}")
    print(f"Median: {median_value}")
    print(f"Mean Absolute Error (MAE): {mae} ({mae/mean_value*100:.1f}%)")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse} ({rmse/mean_value*100:.1f}%)")
    print(f"R-squared (R²): {r2}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")
    print(f"Median Absolute Error: {median_ae} ({median_ae/median_value*100:.1f}%)")
    print(f"Explained Variance Score: {evs}")
    print(f'Bias: {bias:.2f}')

    if metric_type:
        metrics_dict = {
            "mean": ["Mean", mean_value],
            "median": ["Median", median_value],
            "mae": ["Mean Absolute Error (MAE)", mae],
            "mse": ["Mean Squared Error (MSE)", mse],
            "rmse": ["Root Mean Squared Error (RMSE)", rmse],
            "r2": ["R-squared (R²)", r2],
            "mape": ["Mean Absolute Percentage Error (MAPE)", mape],
            "median_ae": ["Median Absolute Error", median_ae],
            "evs": ["Explained Variance Score", evs]
        }

        metric_name = metrics_dict[metric_type][0]
        metric_value = metrics_dict[metric_type][1]
        print(f"Metric: {metric_name}")
        print(f"Metric value: {metric_value}")


    #   PLOTS
    # Plot actual vs. predicted values with a 1:1 line and metrics
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_pred, alpha=0.3, s=100)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Measured TLA, cm²', fontsize=16)
    plt.ylabel('Predicted TLA, cm²', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Actual Vs. Predicted TLA', fontsize=16)
    plt.axis('square')
    plt.xlim([min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())])
    plt.ylim([min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())])

    # Fit a linear regression model between y_test and y_pred
    linear_regressor = LinearRegression()
    y_pred_for_lin = y_pred.reshape(-1, 1)  # Reshape y_pred for fitting the linear model
    linear_regressor.fit(y_pred_for_lin, y_test)

    # Make predictions using the linear regression model
    y_pred_linear = linear_regressor.predict(y_pred_for_lin)

    # Display text
    if regression_model == 'linear' or not metric_type:
        plt.text(0.05, 0.95, f'R² = {r2:.2f}', transform=plt.gca().transAxes,
                 fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    else:
        plt.text(0.05, 0.95, f'R² = {r2:.2f}\n{metric_name} = {metric_value:.0f} cm²', transform=plt.gca().transAxes,
                 fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    plt.text(0.05, 0.85, f'3D Reconstruction Algorithm: {formatted_algorithm_name}\n{parameter_value_type[algorithm_name]} Value: {formatted_parameter_value}\nRegression Model: {model_names[regression_model]}', transform=plt.gca().transAxes,
             fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # plt.show()
    plt.savefig(os.path.join(folder_paths['plots'], '20250116', f'pred_{metric_type}.png'), dpi=600)

    # # Plot linear fit
    # # Calculate R² for the linear regression model
    # linfit_r2 = r2_score(y_test, y_pred_linear)
    #
    # # Plot actual vs predicted values with the linear fit line
    # plt.figure(figsize=(10, 10))
    # plt.scatter(y_test, y_pred, alpha=0.3)
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=2, color='blue')
    # plt.xlabel('Measured TLA, Cm²')
    # plt.ylabel('Predicted TLA, Cm²')
    # plt.title('Actual Vs. Predicted Measured TLA')
    # plt.axis('square')
    # plt.xlim([min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())])
    # plt.ylim([min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())])
    # plt.text(0.05, 0.75, f'Linear Fit R²: {round(linfit_r2, 2)}', transform=plt.gca().transAxes,
    #          fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    #
    # if regression_model == 'linear' or not metric_type:
    #     plt.text(0.05, 0.95, f'R² = {r2:.2f}', transform=plt.gca().transAxes,
    #              fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    # else:
    #     plt.text(0.05, 0.95, f'R² = {r2:.2f}\n{metric_name} = {metric_value:.0f} cm²', transform=plt.gca().transAxes,
    #                  fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    # plt.text(0.05, 0.85, f'3D Reconstruction Algorithm: {formatted_algorithm_name}\n{parameter_value_type[algorithm_name]} Value: {formatted_parameter_value}\nRegression Model: {model_names[regression_model]}', transform=plt.gca().transAxes,
    #          fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    # plt.show()
    # # plt.savefig(os.path.join(folder_paths['plots'], '20241031', 'pred.png'), dpi=600)

    # Plot residuals
    # Calculate residuals
    residuals = y_test - y_pred

    # Create the figure
    plt.figure(figsize=(10, 10))
    plt.scatter(y_pred, residuals, alpha=0.3, s=100)
    plt.axhline(0, color='r', linestyle='--', lw=2)  # Line at y=0 to indicate no residual
    plt.xlabel('Predicted TLA, cm²', fontsize=16)
    plt.ylabel('Residuals (Measured - Predicted), cm²', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Residuals Plot', fontsize=16)
    plt.axis('square')
    plt.xlim([min(y_pred.min(), y_test.min()), max(y_pred.max(), y_test.max())])
    plt.ylim([residuals.min() * 1.1, residuals.max() * 1.1])

    # Display text with model information
    if regression_model == 'linear' or not metric_type:
        plt.text(0.05, 0.95, f'R² = {r2:.2f}', transform=plt.gca().transAxes,
                 fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    else:
        plt.text(0.05, 0.95, f'R² = {r2:.2f}\n{metric_name} = {metric_value:.0f} cm²', transform=plt.gca().transAxes,
                 fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    plt.text(0.05, 0.80,
         f'3D Reconstruction Algorithm: {formatted_algorithm_name}\n{parameter_value_type[algorithm_name]} Value: {formatted_parameter_value}\nRegression Model: {model_names[regression_model]}',
             transform=plt.gca().transAxes, fontsize=16, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Save the residual plot
    # plt.show()
    plt.savefig(os.path.join(folder_paths['plots'], '20250116', f'residuals_{metric_type}.png'), dpi=600)

    # Plot feature importance
    if regression_model != "linear" and regression_model != "mlp":
        # Plot feature importance as a pie chart
        importance = model.get_score(importance_type='weight')
        importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        features = [format_text(item[0]) for item in importance]
        scores = [item[1] for item in importance]

        plt.figure()
        plt.pie(scores, labels=features, autopct='%1.1f%%', startangle=140)
        plt.title('Feature importance analysis of the best yielding XGBoost model')
        plt.show()
        # plt.savefig(os.path.join(folder_paths['plots'], '20241031', 'feat.png'), dpi=400)


    # Plot histogram of measured leaf area
    # Map numerical experiment numbers to categorical labels
    experiment_mapping = {1: 'Exp. #1', 2: 'Exp. #2'}
    df_original['experiment_number'] = df_original['experiment_number'].map(experiment_mapping)

    # Convert experiment_number to a categorical variable with the desired order
    df_original['experiment_number'] = pd.Categorical(df_original['experiment_number'],
                                                      categories=['Exp. #2', 'Exp. #1'], ordered=True)

    # Plot a histogram
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df_original, x='measured_leaf_area',
                 hue='experiment_number',
                 kde=True,
                 palette='deep',
                 alpha=0.7)
    # Add labels and title
    plt.xlabel('Measured TLA, cm²', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.title('Histogram of Measured TLA by Exp. #', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # Add mean and median lines
    plt.axvline(mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.0f} cm²')
    plt.axvline(median_value, color='b', linestyle='-', label=f'Median: {median_value:.0f} cm²')
    # Add legend
    plt.legend(labels=['Exp. #1', 'Exp. #2', f'Mean: {mean_value:.0f} cm²', f'Median: {median_value:.0f} cm²'],
               fontsize=16)

    # Show plot
    # plt.show()
    plt.savefig(os.path.join(folder_paths['plots'], '20250116', 'dist.png'), dpi=600)