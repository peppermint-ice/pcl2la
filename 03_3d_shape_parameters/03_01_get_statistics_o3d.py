import os
import pandas as pd
import leaf_area as la
import re
import open3d as o3d
import sys
from config import paths


if __name__ == '__main__':
    # Get command line arguments
    approach = sys.argv[1]
    value = sys.argv[2]

    folder_paths = paths.get_paths()
    meshes_folder_path = folder_paths["meshes"]
    pcl_folder_path = folder_paths["corrected"]
    approach_folder_path = os.path.join(meshes_folder_path, approach)
    value_folder_path = os.path.join(approach_folder_path, value)
    print('o3d')
    print('folders set')
    print(approach_folder_path)
    print(value_folder_path)

    # Check if the value_folder_path is a directory
    if os.path.isdir(value_folder_path):
        data = []

        for ply in os.listdir(value_folder_path):
            print('start analysis')
            try:
                # Open mesh and point cloud files
                pcl_file_path = os.path.join(pcl_folder_path, ply)
                ply_path = os.path.join(value_folder_path, ply)

                pcl = o3d.io.read_point_cloud(pcl_file_path)
                mesh = o3d.io.read_triangle_mesh(ply_path)

                # Run the analysis
                total_volume = la.calculate_watertight_volume(mesh)
                stats = la.calculate_shape_parameters(pcl, mesh, total_volume)

                # Add data about the files to the analysis results
                #       1. Get measured leaf area data
                match = re.search(r'(\d+p\d+\.)', ply)
                #       2. Add parameters to the file
                if match:
                    measured_leaf_area = float(match.group().replace('p', '.')[:-1])
                    print('Measured leaf area:', measured_leaf_area)
                    stats['measured_leaf_area'] = measured_leaf_area

                    data.append(stats)
                    print("")
            except TypeError:
                print('TypeError: something is not good')

        # Convert data to DataFrame and transpose it
        df = pd.DataFrame(data).transpose()

        # Save the transposed DataFrame to CSV
        csv_folder_path = folder_paths["ready_for_training"]
        simples_file_name = approach + "_" + value + "_o3d_simple_noElim.csv"
        simples_path = os.path.join(csv_folder_path, simples_file_name)
        df.to_csv(simples_path, index=False)
        print("Files saved here: ", simples_path)
