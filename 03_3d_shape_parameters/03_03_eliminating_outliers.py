import os
import leaf_area as la
from config import paths


if __name__ == '__main__':
    folder_paths = paths.get_paths()
    csv_folder_path = folder_paths["ready_for_training"]

    # Process each CSV file in the folder
    for filename in os.listdir(csv_folder_path):
        if filename.endswith("_noElim.csv"):
            import_csv_path = os.path.join(csv_folder_path, filename)
            export_csv_path = os.path.join(csv_folder_path, filename.replace("_noElim.csv", "_elim.csv"))
            la.eliminate_outliers_and_scale(import_csv_path, export_csv_path)