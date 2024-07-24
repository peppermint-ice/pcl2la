import os
import pandas as pd
import leaf_area as la
import re
import open3d as o3d
import trimesh
import numpy as np
from config import paths

if __name__ == '__main__':
    # Define the folder paths
    folder_paths = paths.get_paths()
    value_folder_path = r'C:\Users\dusen\Documents\PhD\plys\meshes\marching_cubes\0.8'
    pcl_folder_path = r'G:\My Drive\Dmitrii - Ph.D Thesis\Frost room Experiment Data\LA\plys\corrected_ready_to_execute'

    # Check if the value_folder_path is a directory
    if os.path.isdir(value_folder_path):
        data = []

        for ply in os.listdir(value_folder_path):
            print('start analysis')
            try:
                # Open mesh and point cloud files
                pcl_file_path = os.path.join(pcl_folder_path, ply)
                ply_path = os.path.join(value_folder_path, ply)

                # Read the mesh file
                mesh = o3d.io.read_triangle_mesh(ply_path)

                # Run the analysis
                total_volume = la.calculate_watertight_volume(mesh)
                stats = la.calculate_shape_parameters(pcl_file_path, ply_path, total_volume)

                # Add data about the files to the analysis results
                #       1. Get measured leaf area data
                match = re.search(r'(\d+p\d+\.)', ply)
                #       2. Add parameters to the file
                if match:
                    measured_leaf_area = float(match.group().replace('p', '.')[:-1])
                    print('Measured leaf area:', measured_leaf_area)
                    stats['measured_leaf_area'] = measured_leaf_area
                    data.append(stats)
                    print(stats)
                    print("")
            except Exception as e:
                print(f'Error: {e.__class__.__name__}, {e}')

        # Convert data to DataFrame
        df = pd.DataFrame(data)
        print(df)

        # Save the DataFrame to CSV
        csv_folder_path = folder_paths["ready_for_training_2"]
        simples_file_name = "marching_cubes_0.8_o3d_simple_noElim.csv"
        simples_path = os.path.join(csv_folder_path, simples_file_name)
        # df.to_csv(simples_path, index=False)
        # print("Files saved here: ", simples_path)
