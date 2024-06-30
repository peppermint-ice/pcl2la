import os
import pandas as pd
import leaf_area as la
import re
import open3d as o3d
import trimesh
import numpy as np
from config import paths

def get_year_from_filename(filename):
    year_prefix = filename[:6]
    if year_prefix.endswith('23'):
        return 2023
    elif year_prefix.endswith('24'):
        return 2024
    else:
        return None

def calculate_watertight_volume(shape):
    vertices = np.asarray(shape.vertices)
    triangles = np.asarray(shape.triangles)
    # Create a Trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    total_volume = 0
    for i in mesh.split():
        if i.is_watertight:
            print(i.volume)
            total_volume += abs(i.volume)
    return total_volume

def calculate_shape_parameters(point_cloud_file_path, mesh_file_path, total_volume):
    point_cloud = o3d.io.read_point_cloud(point_cloud_file_path)
    point_cloud_array = np.asarray(point_cloud.points)
    # Get dimensions (height, length, width)
    dimensions = np.ptp(point_cloud_array, axis=0)

    # Calculate surface area
    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    surface_area = mesh.get_surface_area()

    # Calculate aspect ratio
    try:
        aspect_ratio = np.max(dimensions) / np.min(dimensions)
    except ZeroDivisionError:
        print("Can't calculate aspect ratio. Division by zero. Aspect Ratio set to 0")
        aspect_ratio = 0

    # Calculate elongation
    try:
        elongation = (np.max(dimensions) / np.median(dimensions)) - 1
    except ZeroDivisionError:
        print("Can't calculate elongation. Division by zero.")
        elongation = 0

    # Calculate flatness
    try:
        flatness = (np.min(dimensions) / np.median(dimensions)) - 1
    except ZeroDivisionError:
        print("Can't calculate flatness. Division by zero.")
        flatness = 0

    # Calculate sphericity for the entire alpha shape
    try:
        sphericity = (np.pi ** (1 / 3)) * ((6 * total_volume) ** (2 / 3)) / surface_area
    except ZeroDivisionError:
        print("Can't calculate sphericity. Division by zero.")
        sphericity = 0

    # Calculate compactness for the entire alpha shape
    try:
        compactness = (36 * np.pi * total_volume ** 2) ** (1 / 3) / surface_area
    except ZeroDivisionError:
        print("Can't calculate compactness. Division by zero.")
        compactness = 0

    # Calculate number of independent components
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    components = 0
    for i in trimesh_mesh.split():
        components += 1

    # Get the number of points inside the alpha shape
    num_points_inside = len(point_cloud_array)
    try:
        point_density = num_points_inside / total_volume
    except ZeroDivisionError:
        print("Can't calculate point density. Division by zero.")
        point_density = 0

    # Store parameters for the entire alpha shape
    parameters = {
        'height': dimensions[2],  # Assuming z is height
        'length': dimensions[0],  # Assuming x is length
        'width': dimensions[1],   # Assuming y is width
        'volume': total_volume,
        'surface_area': surface_area,
        'aspect_ratio': aspect_ratio,
        'components_number': components,
        'sphericity': sphericity,
        'compactness': compactness,
        'point_density': point_density,
    }
    return parameters

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
                total_volume = calculate_watertight_volume(mesh)
                stats = calculate_shape_parameters(pcl_file_path, ply_path, total_volume)

                # Add data about the files to the analysis results
                #       1. Get measured leaf area data
                match = re.search(r'(\d+p\d+\.)', ply)
                #       2. Add parameters to the file
                if match:
                    measured_leaf_area = float(match.group().replace('p', '.')[:-1])
                    print('Measured leaf area:', measured_leaf_area)
                    stats['measured_leaf_area'] = measured_leaf_area

                    # Get year from filename
                    year = get_year_from_filename(ply)
                    stats['Year'] = year

                    data.append(stats)
                    print("")
            except Exception as e:
                print(f'Error: {e.__class__.__name__}, {e}')

        # Convert data to DataFrame
        df = pd.DataFrame(data)

        # Save the DataFrame to CSV
        csv_folder_path = folder_paths["ready_for_training_2"]
        simples_file_name = "marching_cubes_0.8_o3d_simple_noElim.csv"
        simples_path = os.path.join(csv_folder_path, simples_file_name)
        df.to_csv(simples_path, index=False)
        print("Files saved here: ", simples_path)
