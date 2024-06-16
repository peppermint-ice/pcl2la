from config import paths
import os
import pandas as pd

# Get path
folder_paths = paths.get_paths()

# Set folder
csv_folder_path = folder_paths["results"]
csv_export_path = os.path.join(folder_paths["combined"], "combine_ML10.csv")
csvs = os.listdir(csv_folder_path)
print(csv_export_path)

df = pd.DataFrame()

for file in csvs:
    file_path = os.path.join(csv_folder_path, file)
    if os.path.isfile(file_path) and file_path.lower().endswith('.csv'):
        df_current = pd.read_csv(file_path)

        df = pd.concat([df, df_current], ignore_index=True)

df.to_csv(csv_export_path, index=False)