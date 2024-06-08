import leaf_area as la
from config import paths
import os
import re
import pandas as pd
import time
import sys

# Extract start and end indices from command-line arguments
start_index = int(sys.argv[1])
end_index = int(sys.argv[2])


# Get path
folder_paths = paths.get_paths()

# Set folders
corrected_folder_path = folder_paths["corrected"]
poisson_folder_path = folder_paths["poissons"]


plys = os.listdir(corrected_folder_path)


# Select desired poisson values

poisson_values = [8, 9, 10, 11, 12, 13, 14, 15]

# Start time measurement
start_time = time.time()

for poisson_value in poisson_values:
    df = pd.DataFrame()
    for file in plys[start_index - 1:end_index]:
        ply_file_path = os.path.join(corrected_folder_path, file)
        print("poisson value: " + str(poisson_value))
        if os.path.isfile(ply_file_path) and ply_file_path.lower().endswith('.ply'):
            # Set up iteration start time
            iteration_time = time.time()

            try:
                # Open a ply
                pcl = la.open_ply_file(ply_file_path)

                # Define export path
                value_folder_path = os.path.join(poisson_folder_path, str(poisson_value))
                os.makedirs(value_folder_path, exist_ok=True)
                mesh_export_path = os.path.join(value_folder_path, file)

                # Create poisson shapes
                poisson_shape = la.create_poisson_shape(ply_file_path, poisson_value, mesh_export_path)
                total_volume = la.calculate_watertight_volume(poisson_shape)

                # Measure time taken for this iteration
                iteration_time = time.time() - iteration_time
                print("Time taken for this iteration: " + str(iteration_time) + " seconds")
            except IndexError:
                print("Index error. The poisson is not good")
                continue


# Total time taken for the loop
total_time = time.time() - start_time
print("Total time taken: " + str(total_time) + " seconds")