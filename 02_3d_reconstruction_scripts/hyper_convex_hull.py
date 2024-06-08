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

# Set import path
folder_paths = paths.get_paths()

# Set export folders
corrected_folder_path = folder_paths["corrected"]
convex_hull_folder_path = folder_paths["convex_hull"]


plys = os.listdir(corrected_folder_path)
df = pd.DataFrame()

# Start time measurement
start_time = time.time()

for file in plys[start_index-1:end_index]:
    ply_file_path = os.path.join(corrected_folder_path, file)
    if os.path.isfile(ply_file_path) and ply_file_path.lower().endswith('.ply'):
        # Set up iteration start time
        iteration_time = time.time()
        print(ply_file_path)

        # Open a ply
        pcl = la.open_ply_file(ply_file_path)

        # Define export path
        mesh_export_path = os.path.join(convex_hull_folder_path, file)

        # Create convex_hull shapes
        radii_value = 5  # Adjust convex_hull depth as needed
        convex_hull_shape = la.create_convex_hull_shape(ply_file_path, mesh_export_path)
        total_volume = la.calculate_watertight_volume(convex_hull_shape)

        # Measure time taken for this iteration
        iteration_time = time.time() - iteration_time
        print("Time taken for this iteration: " + str(iteration_time) + " seconds")

# Total time taken for the loop
total_time = time.time() - start_time
print("Total time taken: " + str(total_time) + " seconds")