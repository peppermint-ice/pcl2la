import os
import pandas as pd
from config import paths

# Set folder paths
folder_paths = paths.get_paths()
file_path = os.path.join(folder_paths['combined'], "combine_ML_240724_1608.csv")
export_path = os.path.join(folder_paths['combined'], "2507data.csv")

# Load the CSV file
df = pd.read_csv(file_path)

# Select rows where elimination_status is 'noElim'
filtered_df = df[df['elimination_status'] == 'noElim']

# Save the filtered data to a new file
filtered_df.to_csv(export_path, index=False)
print(f"Filtered data saved to {export_path}")