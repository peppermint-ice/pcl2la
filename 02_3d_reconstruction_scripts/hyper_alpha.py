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
alpha_folder_path = folder_paths["alphas"]


plys = os.listdir(corrected_folder_path)


# Select desired alpha values

alpha_values = [0.001, 0.005, 0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 1.75, 2, 3, 4, 5, 7.5, 10, 15, 20, 25, 50, 100, 1000]

# Start time measurement
start_time = time.time()

for alpha_value in alpha_values:
    df = pd.DataFrame()
    for file in plys[start_index - 1:end_index]:
        ply_file_path = os.path.join(corrected_folder_path, file)
        print("Alpha value: " + str(alpha_value))
        if os.path.isfile(ply_file_path) and ply_file_path.lower().endswith('.ply'):
            # Set up iteration start time
            iteration_time = time.time()

            try:
                # Open a ply
                pcl = la.open_ply_file(ply_file_path)

                # Define export path
                value_folder_path = os.path.join(alpha_folder_path, str(alpha_value))
                os.makedirs(value_folder_path, exist_ok=True)
                mesh_export_path = os.path.join(value_folder_path, file)

                # Create alpha shapes
                print('was', ply_file_path)
                print('is', mesh_export_path)
                alpha_shape = la.create_alpha_shape(ply_file_path, alpha_value, mesh_export_path)

                # Measure time taken for this iteration
                iteration_time = time.time() - iteration_time
                print("Time taken for this iteration: " + str(iteration_time) + " seconds")

            except IndexError:
                print("Index error. The alpha is not good")
                continue

# Total time taken for the loop
total_time = time.time() - start_time
print("Total time taken: " + str(total_time) + " seconds")