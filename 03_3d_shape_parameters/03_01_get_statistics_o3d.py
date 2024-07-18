import os
import pandas as pd
import leaf_area as la
import re
import open3d as o3d
import sys
from config import paths
import traceback


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
                print('files read successfully')

                # Run the analysis
                total_volume = la.calculate_watertight_volume(mesh)
                print('volume calculated')
                print('calculating stats')
                stats = la.calculate_shape_parameters(pcl, mesh, total_volume)
                print('stats calculated')

                # Add LA and year from file name to the analysis results
                # Pattern: (date)_(everythingelse)_(LA).ply
                pattern = r'(\d*)_(\w*)_(\d*p?\d*)\.ply'
                match = re.search(pattern, ply)
                #       Get measured leaf area and year data. Add parameters to the file
                if match:
                    measured_leaf_area = float(match.group(3).replace('p', '.'))
                    print('Measured leaf area:', measured_leaf_area)
                    stats['measured_leaf_area'] = measured_leaf_area
                    if match.group(1).endswith("23"):
                        stats['experiment_number'] = 1
                    elif match.group(1).endswith("24"):
                        stats['experiment_number'] = 2

                    data.append(stats)
                    print("info retrieved from file name")
                    print("")
            except TypeError:
                print('TypeError: something is not good')
                traceback.print_exc()

        # Convert data to DataFrame
        df = pd.DataFrame(data)

        # Save the transposed DataFrame to CSV
        csv_folder_path = folder_paths["ready_for_training"]
        simples_file_name = approach + "_" + value + "_o3d_simple_noElim.csv"
        simples_path = os.path.join(csv_folder_path, simples_file_name)
        df.to_csv(simples_path, index=False)
        print("Files saved here: ", simples_path)
