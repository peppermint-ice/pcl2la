import pandas as pd
from config import paths
import os

folder_paths = paths.get_paths()
combined_folder_path = folder_paths['combined']
data = pd.read_csv(os.path.join(combined_folder_path, '2507data.csv'))
r2_df = pd.read_csv(os.path.join(combined_folder_path, 'combine_ML_060824_1807_filtered_r2.csv'))
rmse_df = pd.read_csv(os.path.join(combined_folder_path, 'combine_ML_060824_1807_filtered_rmse.csv'))

# Function to get the best R2 and RMSE
def get_best_values(df, value_col, agg_func):
    return df.loc[df.groupby(['algorithm_name', 'regression_model'])[value_col].transform(agg_func) == df[value_col]]


# Get best R2 and RMSE values
best_r2 = get_best_values(r2_df, 'R2_global_test', 'max')
best_rmse = get_best_values(rmse_df, 'RMSE_global_test', 'min')


# Filter for each year_param and merge the best values
def filter_and_merge(df, year_param):
    filtered_df = df[df['byyear'] == year_param]
    best_r2_filtered = get_best_values(filtered_df, 'R2_global_test', 'max')
    best_rmse_filtered = get_best_values(filtered_df, 'RMSE_global_test', 'min')
    return best_r2_filtered, best_rmse_filtered


# Get best values for each year_param
best_r2_byyear, best_rmse_byyear = filter_and_merge(r2_df, 'byyear')
best_r2_byyearreversed, best_rmse_byyearreversed = filter_and_merge(r2_df, 'byyearreveresed')
best_r2_noyear, best_rmse_noyear = filter_and_merge(r2_df, 'noyear')

# Create the table
table_columns = [
    'Reconstruction algorithm', 'Regression model',
    'Whole dataset Best R2', 'Whole dataset Best RMSE',
    'Train: Exp #1, test: Exp #2 Best R2', 'Train: Exp #1, test: Exp #2 Best RMSE',
    'Train: Exp #2, test: Exp #1 Best R2', 'Train: Exp #2, test: Exp #1 Best RMSE'
]

# Initialize the table
results_table = pd.DataFrame(columns=table_columns)


# Function to add rows to the table
def add_rows_to_table(best_r2, best_rmse, best_r2_byyear, best_rmse_byyear, best_r2_byyearreversed,
                      best_rmse_byyearreversed):
    for _, row in best_r2.iterrows():
        reconstruction_algorithm = row['algorithm_name']
        regression_model = row['regression_model']

        # Get the corresponding RMSE value
        rmse_value = best_rmse[
            (best_rmse['algorithm_name'] == reconstruction_algorithm) &
            (best_rmse['regression_model'] == regression_model)
            ]['RMSE_global_test'].values[0]

        # Get the corresponding R2 and RMSE values for 'byyear' and 'byyearreversed'
        byyear_r2_value = best_r2_byyear[
            (best_r2_byyear['algorithm_name'] == reconstruction_algorithm) &
            (best_r2_byyear['regression_model'] == regression_model)
            ]['R2_global_test'].values[0]

        byyear_rmse_value = best_rmse_byyear[
            (best_rmse_byyear['algorithm_name'] == reconstruction_algorithm) &
            (best_rmse_byyear['regression_model'] == regression_model)
            ]['RMSE_global_test'].values[0]

        byyearreversed_r2_value = best_r2_byyearreversed[
            (best_r2_byyearreversed['algorithm_name'] == reconstruction_algorithm) &
            (best_r2_byyearreversed['regression_model'] == regression_model)
            ]['R2_global_test'].values[0]

        byyearreversed_rmse_value = best_rmse_byyearreversed[
            (best_rmse_byyearreversed['algorithm_name'] == reconstruction_algorithm) &
            (best_rmse_byyearreversed['regression_model'] == regression_model)
            ]['RMSE_global_test'].values[0]

        # Add the row to the table
        results_table.loc[len(results_table)] = [
            reconstruction_algorithm, regression_model,
            row['R2_global_test'], rmse_value,
            byyear_r2_value, byyear_rmse_value,
            byyearreversed_r2_value, byyearreversed_rmse_value
        ]


# Add rows to the table
add_rows_to_table(best_r2, best_rmse, best_r2_byyear, best_rmse_byyear, best_r2_byyearreversed,
                  best_rmse_byyearreversed)

# Display the results table
print(results_table.to_string())

# Save the results table to an Excel file
# results_table.to_excel('results_table.xlsx', index=False)