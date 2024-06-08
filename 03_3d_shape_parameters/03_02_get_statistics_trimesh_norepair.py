import os
import pandas as pd
import leaf_area as la
import re
import sys
from config import paths

# Get command line arguments
approach = sys.argv[1]
value = sys.argv[2]

folder_paths = paths.get_paths()
folder_path = folder_paths["meshes"]
approach_folder_path = os.path.join(folder_path, approach)
value_folder_path = os.path.join(approach_folder_path, value)
print('folders set')
print(approach_folder_path)
print(value_folder_path)

# Check if the value_folder_path is a directory
if os.path.isdir(value_folder_path):
    data = []

    for ply in os.listdir(value_folder_path):
        print('start analysis')
        try:
            # Run analysis
            ply_path = os.path.join(value_folder_path, ply)
            print(ply_path)
            stats = la.calculate_statistics_trimesh(ply_path, repair=False)
            # Add data about the files to the analysis results
            #       1. Get measured leaf area data
            match = re.search(r'(\d+p\d+\.)', ply)
            #       2. Add parameters to the file
            if match:
                measured_leaf_area = float(match.group().replace('p', '.')[:-1])
                print('Measured leaf area:', measured_leaf_area)
                stats['measured_leaf_area'] = measured_leaf_area
                stats['parameter_name'] = approach
                stats['parameter_value'] = value

                data.append(stats)
                print("")
        except TypeError:
            print('TypeError: something is not good')

    # Convert data to DataFrame and transpose it
    df = pd.DataFrame(data).transpose()

    # Save the transposed DataFrame to CSV
    csv_folder_path = folder_paths["ready_for_training"]
    simples_file_name = approach + "_" + value + "_trimesh_simple.csv"
    simples_path = os.path.join(csv_folder_path, simples_file_name)
    df.to_csv(simples_path, index=False)
    print("Files saved here: ", simples_path)
