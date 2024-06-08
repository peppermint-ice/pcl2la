import leaf_area as la
from config import paths
import os
import time

# The whole procedure

# Set import path
folder_paths = paths.get_paths()
# Set export folders
ply_folder_path = folder_paths["plys"]
clipped_export_folder_path = folder_paths["clipped_cubes"]
red_from_clipped_export_folder_path = folder_paths["red_from_clipped_cubes"]
clipped_clustered_export_folder_path = folder_paths["clustered_cubes"]
clipped_clustered_green_export_folder_path = folder_paths["green_cubes"]
clipped_clustered_red_export_folder_path = folder_paths["red_cubes"]
plys = os.listdir(ply_folder_path)

# Start time measurement
start_time = time.time()

for file in plys:
    ply_file_path = os.path.join(ply_folder_path, file)
    if os.path.isfile(ply_file_path) and ply_file_path.lower().endswith('.ply'):
        # Set up iteration start time
        iteration_time = time.time()

        # Set export paths
        clipped_export_file_path = os.path.join(folder_paths['clipped_cubes'], file)
        red_from_clipped_export_file_path = os.path.join(folder_paths['red_from_clipped_cubes'], file)
        clipped_clustered_export_file_path = os.path.join(folder_paths['clustered_cubes'], file)
        clipped_clustered_green_export_file_path = os.path.join(folder_paths['green_cubes'], file)
        clipped_clustered_red_export_file_path = os.path.join(folder_paths['red_cubes'], file)
        print(ply_file_path)
        print(clipped_export_file_path)
        print(clipped_clustered_export_file_path)
        print(clipped_clustered_green_export_file_path)
        print(clipped_clustered_red_export_file_path)

        # Open a file
        pcl = la.open_ply_file(ply_file_path)

        # Cropping
        vertices, _ = la.extract_camera_points(pcl)
        cube = la.create_a_cube(vertices)
        vertices, colors = la.crop_a_point_cloud(pcl, cube, 'cube')
        cropped_pcl = la.export_ply_file(vertices, colors, clipped_export_file_path)

        # Clustering
        vertices, colors, assigned_clusters, df = la.create_clusters(cropped_pcl)
        clustered_pcl = la.export_ply_file(vertices, colors, clipped_clustered_export_file_path)

        # Color filtering
        vertices, colors = la.green_index(clustered_pcl)
        green_pcl = la.export_ply_file(vertices, colors, clipped_clustered_green_export_file_path)
        vertices, colors = la.red_index(clustered_pcl)
        red_pcl = la.export_ply_file(vertices, colors, clipped_clustered_red_export_file_path)

        # Red from clipped
        vertices, colors = la.red_index(cropped_pcl)
        red_pcl = la.export_ply_file(vertices, colors, red_from_clipped_export_file_path)

        # Measure time taken for this iteration
        iteration_time = time.time() - iteration_time
        print("Time taken for this iteration: " + str(iteration_time) + " seconds")

# Total time taken for the loop
total_time = time.time() - start_time
print("Total time taken: " + str(total_time) + " seconds")