import os
import pandas as pd
import leaf_area as la
import re
import sys
from config import paths


if __name__ == '__main__':
    # Get command line arguments
    approach = sys.argv[1]
    value = sys.argv[2]

    folder_paths = paths.get_paths()
    folder_path = folder_paths["meshes"]
    approach_folder_path = os.path.join(folder_path, approach)
    value_folder_path = os.path.join(approach_folder_path, value)
    print('trimesh')
    print('folders set')
    print(folder_path)
    print(approach_folder_path)
    print(value_folder_path)

    # Check if the value_folder_path is a directory
    if os.path.isdir(value_folder_path):
        simples_data = []
        repaired_data = []

        for ply in os.listdir(value_folder_path):
            print('start analysis')
            try:
                # Run analysis
                ply_path = os.path.join(value_folder_path, ply)
                print(ply_path)
                stats = la.calculate_statistics_trimesh(ply_path, repair=True)
                # Add LA and year from file name to the analysis results
                # Pattern: (date)_(everythingelse)_(LA).ply
                pattern = r'(\d*)_(\w*)_(\d*p?\d*)\.ply'
                match = re.search(pattern, ply)
                #       Get measured leaf area and year data. Add parameters to the file
                if match:
                    measured_leaf_area = float(match.group(3).replace('p', '.'))
                    print('Measured leaf area:', measured_leaf_area)
                    stats[0]['measured_leaf_area'] = measured_leaf_area
                    stats[1]['measured_leaf_area'] = measured_leaf_area
                    if match.group(1).endswith("23"):
                        stats[0]['experiment_number'] = 1
                        stats[1]['experiment_number'] = 1
                    elif match.group(1).endswith("24"):
                        stats[0]['experiment_number'] = 2
                        stats[1]['experiment_number'] = 2

                    simples_data.append(stats[0])
                    repaired_data.append(stats[1])
                    print("")
            except TypeError:
                print('TypeError: something is not good')

        # Convert data to DataFrame
        simples_df = pd.DataFrame(simples_data)
        repaired_df = pd.DataFrame(repaired_data)

        # Save the DataFrames to CSV
        csv_folder_path = folder_paths["ready_for_training"]
        simples_file_name = approach + "_" + value + "_trimesh_simple_noElim.csv"
        simples_path = os.path.join(csv_folder_path, simples_file_name)
        repaired_file_name = approach + "_" + value + "_trimesh_repaired_noElim.csv"
        repaired_path = os.path.join(csv_folder_path, repaired_file_name)
        simples_df.to_csv(simples_path, index=False)
        repaired_df.to_csv(repaired_path, index=False)
        print("Files saved here: ", simples_path)
        print("Files saved here: ", repaired_path)