import pandas as pd
from config import paths
import os

folder_paths = paths.get_paths()
combined_folder_path = folder_paths['combined']
data = pd.read_csv(os.path.join(combined_folder_path, '2507data.csv'))

data = data[data['byyear'] == 'byyear']

# Group by algorithm and regression model and find the best R2
best_scores = data.groupby(['algorithm_name', 'regression_model']).apply(
    lambda df: df.loc[df['R2_global_test'].idxmax(), ['R2_global_test']]
).reset_index()

# Rename columns for clarity
best_scores.columns = ['Reconstruction Algorithm', 'Regression Model', 'Best R2']

# Display the resulting table
print(best_scores)
print('')

# Group by algorithm and regression model and find the best RMSE
best_scores = data.groupby(['algorithm_name', 'regression_model']).apply(
    lambda df: df.loc[df['RMSE_global_test'].idxmin(), ['RMSE_global_test']]
).reset_index()

# Rename columns for clarity
best_scores.columns = ['Reconstruction Algorithm', 'Regression Model', 'Best RMSE']

# Display the resulting table
print(best_scores)