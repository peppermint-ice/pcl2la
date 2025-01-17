import os
import pandas as pd
from config import paths

# Set paths
folder_paths = paths.get_paths()
combined_folder_path = folder_paths["combined"]
file_name = "combine_ML_060824_1807.csv"
file_path = os.path.join(combined_folder_path, file_name)

# Set the parameter to filter by
filter_parameter = 'R2_global_test'

# Load the dataset
df = pd.read_csv(file_path)

# Group by 'algorithm_name' and 'parameter_value', then select the row with the highest 'R2_global_test' for each group
df_filtered = df.loc[df.groupby(['algorithm_name', 'parameter_value', 'regression_model', 'byyear'])[filter_parameter].idxmax()]

# Save the filtered dataframe to a new CSV file
output_file_name = file_name[:-4] + "_filtered_r2.csv"
output_path = os.path.join(combined_folder_path, output_file_name)
df_filtered.to_csv(output_path, index=False)

print(f"Filtered data saved to {output_path}")
