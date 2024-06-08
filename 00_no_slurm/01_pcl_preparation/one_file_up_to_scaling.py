import leaf_area as la
from config import paths
import os


# Set import path
folder_paths = paths.get_paths()


# Set export folders
file = r'240124_53_1_867p58.ply'
ply_folder_path = folder_paths["plys"]
clipped_export_folder_path = folder_paths["clipped_cubes"]
red_from_clipped_export_folder_path = folder_paths["red_from_clipped_cubes"]
clipped_clustered_export_folder_path = folder_paths["clustered_cubes"]
clipped_clustered_green_export_folder_path = folder_paths["green_cubes"]
clipped_clustered_red_export_folder_path = folder_paths["red_cubes"]


# Set paths to files
clipped_export_file_path = os.path.join(clipped_export_folder_path, file)
red_from_clipped_export_file_path = os.path.join(red_from_clipped_export_folder_path, file)
clipped_clustered_export_file_path = os.path.join(clipped_clustered_export_folder_path, file)
clipped_clustered_green_export_file_path = os.path.join(clipped_clustered_green_export_folder_path, file)
clipped_clustered_red_export_file_path = os.path.join(clipped_clustered_red_export_folder_path, file)


# Open a file
ply_file_path = os.path.join(ply_folder_path, file)
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