import os
import leaf_area as la
from config import paths
import pickle

if __name__ == '__main__':
    folder_paths = paths.get_paths()
    csv_folder_path = folder_paths["ready_for_training"]
    scalers_folder_path = folder_paths["scalers"]

    # Process each CSV file in the folder
    for filename in os.listdir(csv_folder_path):
        if filename.endswith("_noElim.csv"):
            import_csv_path = os.path.join(csv_folder_path, filename)
            export_csv_path = os.path.join(csv_folder_path, filename.replace("_noElim.csv", "_elim.csv"))
            scaler = la.eliminate_outliers_and_scale(import_csv_path, export_csv_path)
            scaler_name = filename.replace("_noElim.csv", "_scaler.pkl")
            scaler_export_path = os.path.join(scalers_folder_path, scaler_name)
            with open(scaler_export_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Scaler saved to: {scaler_export_path}")