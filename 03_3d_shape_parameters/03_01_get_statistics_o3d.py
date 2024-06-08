import os
import pandas as pd
import leaf_area as la
import re


folder_path = 'C:/Users/dusen/Documents/PhD/plys/meshes'
pcl_folder_path = 'G:/My Drive/Dmitrii - Ph.D Thesis/Frost room Experiment Data/LA/plys/corrected_ready_to_execute'
approaches = os.listdir(folder_path)

simples = pd.DataFrame()
repaired = pd.DataFrame()

# Go through folders
for approach in approaches:
    print(approach)
    approach_folder_path = os.path.join(folder_path, approach)
    for value in os.listdir(approach_folder_path):
        value_folder_path = os.path.join(approach_folder_path, value)
        # Check that it's a folder and not a random file
        if os.path.isdir(value_folder_path):
            for ply in os.listdir(value_folder_path):
                try:
                    # Run analysis
                    pcl_file_path = os.path.join(pcl_folder_path, ply)
                    ply_path = os.path.join(value_folder_path, ply)
                    total_volume = la.calculate_watertight_volume()
                    stats = la.calculate_shape_parameters(ply_path, total_volume)
                    # Add data about the files to the analysis results
                    #       1. Get measured leaf area data
                    match = re.search(r'(\d+p\d+\.)', ply)
                    #       2. Add parameters to the file
                    if match:
                        print('Measured leaf area:', float(match.group().replace('p', '.')[:-1]))
                        stats[0]['measured_leaf_area'] = float(match.group().replace('p', '.')[:-1])
                        stats[0]['parameter_name'] = approach
                        stats[0]['parameter_value'] = value
                        stats[1]['measured_leaf_area'] = float(match.group().replace('p', '.')[:-1])
                        stats[1]['parameter_name'] = approach
                        stats[1]['parameter_value'] = value

                    pd.concat([simples, pd.Series(stats[0])], axis=1, ignore_index=True)
                    pd.concat([repaired, pd.Series(stats[1])], axis=1, ignore_index=True)
                    print("")
                except TypeError:
                    print('TypeError: something is not good')
simples.to_csv('C:/Users/dusen/Documents/PhD/plys/data/combined/o3d_simples.csv')

