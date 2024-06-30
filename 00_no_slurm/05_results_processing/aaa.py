import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os
import pickle

from config import paths

if __name__ == '__main__':
    # Define the specific file paths
    folder_paths = paths.get_paths()
    parameter_name = "marching_cubes"  # Replace with your actual parameter name if different
    parameter_value = "0.8"  # Replace with your actual parameter value if different
    assessment_name = "o3d_simple_noElim"  # Replace with your actual assessment name if different
    repaired = "by"
    eliminated = "year"

    model_filename = f"{parameter_name}_{parameter_value}_{assessment_name}_{repaired}_{eliminated}_by_model_rf.pkl"
    test_filename = f"{parameter_name}_{parameter_value}_{assessment_name}_{repaired}_{eliminated}_by_test_rf.csv"

    model_filepath = os.path.join(folder_paths["models"], model_filename)
    test_filepath = os.path.join(folder_paths["test_sets"], test_filename)

    # Load the test dataset
    test_df = pd.read_csv(test_filepath)

    # Limit the data by measured leaf area (not more than 5000)
    test_df = test_df[test_df['measured_leaf_area'] <= 5000]

    # Separate the features and the target variable
    X_test = test_df.drop(columns=['measured_leaf_area'])
    y_test = test_df['measured_leaf_area']

    # Load the trained model
    with open(model_filepath, 'rb') as f:
        model = pickle.load(f)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    # Plotting predicted vs measured values
    plt.figure(figsize=(8, 8))
    plt.scatter(y_pred, y_test, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Predicted Leaf Area')
    plt.ylabel('Measured Leaf Area')
    plt.title('Predicted vs Measured Leaf Area (2024)')
    plt.xlim([y_test.min(), y_test.max()])
    plt.ylim([y_test.min(), y_test.max()])
    plt.gca().set_aspect('equal', adjustable='box')

    # Add R2 to the plot
    plt.text(0.05, 0.95, f'$R^2 = {r2:.2f}$', transform=plt.gca().transAxes,
             fontsize=14, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    plt.show()
