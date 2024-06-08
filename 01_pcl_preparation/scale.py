import leaf_area as la
from config import paths
import os
import time


folder_paths = paths.get_paths()

circles_folder_path = folder_paths["circles"]
green_folder_path = folder_paths["green_cubes"]
rotated_export_folder_path = folder_paths["rotated_cubes"]
plys = os.listdir(circles_folder_path)

# Start time measurement
start_time = time.time()

for file in plys:
    ply_file_path = os.path.join(circles_folder_path, file)
    green_ply_file_path = os.path.join(green_folder_path, file)
    rotated_export_file_path = os.path.join(rotated_export_folder_path, file)
    if os.path.isfile(ply_file_path) and ply_file_path.lower().endswith('.ply'):
        # Set up iteration start time
        iteration_time = time.time()

        print(ply_file_path)
        print(rotated_export_file_path)
        print(green_ply_file_path)

        pcl = la.open_ply_file(ply_file_path)
        circle_pcl = pcl
        green_pcl = la.open_ply_file(green_ply_file_path)

        scaling_parameters = la.calculate_rotation_and_scaling(circle_pcl)
        vertices, colors = la.transform_point_cloud(green_pcl, scaling_parameters)
        la.export_ply_file(vertices, colors, output_filepath=rotated_export_file_path)

        # Measure time taken for this iteration
        iteration_time = time.time() - iteration_time
        print("Time taken for this iteration: " + str(iteration_time) + " seconds")

# Total time taken for the loop
total_time = time.time() - start_time
print("Total time taken: " + str(total_time) + " seconds")