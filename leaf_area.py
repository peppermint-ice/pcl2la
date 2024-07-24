import numpy as np
import pandas as pd
import open3d as o3d
import mcubes

import pyvista as pv
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from config import paths
from sklearn.preprocessing import StandardScaler

import gc
import os
import re
import random
import trimesh


def open_ply_file(file_path):
    """ This function opens a ply file and returns it as an o3d object """
    point_cloud_data_file = o3d.io.read_point_cloud(file_path)
    return point_cloud_data_file


def export_ply_file(vertices, colors=None, output_filepath=None):
    # Create a new PointCloud object
    point_cloud = o3d.geometry.PointCloud()

    # Set the points and colors of the cropped PointCloud
    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Save the cropped PointCloud to a PLY file in ASCII format
    if output_filepath is not None:
        o3d.io.write_point_cloud(output_filepath, point_cloud, write_ascii=True)

    return point_cloud


def create_clusters(point_cloud_data_file, eps=0.15, min_samples=80):
    """
    This function requires clustering parameters. It runs clustering and returns a big table with the information
    regarding all the points in the ply being assigned to a cluster.
    """

    point_cloud_data_array = np.asarray(point_cloud_data_file.points)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(point_cloud_data_array)
    cluster_labels = dbscan.labels_
    unique_label_names = np.unique(cluster_labels)
    a = 0
    print(unique_label_names)
    for i in unique_label_names:
        for j in cluster_labels:
            if j == i:
                a += 1
        print("Cluster number: " + str(i))
        print("Number of points in the cluster: " + str(a))
        a = 0

    a = 0
    cluster_names_list = []
    cluster_count_list = []
    for i in unique_label_names:
        for j in cluster_labels:
            if j == i:
                a += 1
        if i != -1:
            cluster_names_list.append(i)
            cluster_count_list.append(a)
        a = 0
    clusters_df = pd.DataFrame({"Cluster_Number": cluster_names_list, "Cluster_Count": cluster_count_list})
    clusters_df = clusters_df.sort_values(by="Cluster_Count", ascending=False)
    # print(clusters_df.to_string())

    # Extract the vertices (points) and colors
    vertices = np.asarray(point_cloud_data_file.points)
    colors = np.asarray(point_cloud_data_file.colors)
    # Filter points that are assigned to clusters (not -1)
    clustered_indices = np.where(cluster_labels != -1)[0]
    clustered_points = vertices[clustered_indices]
    original_colors = colors[clustered_indices]
    clustered_colors = cluster_labels

    return clustered_points, original_colors, clustered_colors, clusters_df


def plot_clusters(point_cloud_data_file, cluster_labels):
    """ This function provides a matplotlib 3d plot with a clustered point cloud"""

    # Assuming 'point_cloud_data' contains your 3D point cloud data as a NumPy array
    # Assuming 'cluster_labels' contains the cluster labels assigned by DBSCAN
    point_cloud_data = np.asarray(point_cloud_data_file.points)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get unique cluster labels, excluding -1 (noise label)
    unique_labels = np.unique(cluster_labels[cluster_labels != -1])

    # Define a color map for visualization
    mapped_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    random.shuffle(mapped_colors)

    # Iterate through unique cluster labels and plot points
    for label, color in zip(unique_labels, mapped_colors):
        ax.scatter(point_cloud_data[cluster_labels == label][:, 0],
                   point_cloud_data[cluster_labels == label][:, 1],
                   point_cloud_data[cluster_labels == label][:, 2],
                   c=color, marker='o', s=5, label=f'Cluster {label}')

    # Customize plot labels and legend
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title('DBSCAN Clustering of 3D Point Cloud')
    ax.legend()

    # Show the 3D plot
    plt.show()


def extract_camera_points(point_cloud_data_file):
    """ This function returns an array with points of only one color that represent the camera color"""
    # Creating arrays containing data about points and colors
    vertices = np.asarray(point_cloud_data_file.points)
    colors = np.asarray(point_cloud_data_file.colors)
    # Define the color
    camera_color = [0, 1, 0]
    # Find indices of the dots with the defined color & then extract these points
    camera_indices = np.all(colors == camera_color, axis=1)
    camera_points = vertices[camera_indices]
    camera_points_colors = colors[camera_indices]
    return camera_points, camera_points_colors


def create_a_sphere(camera_points):
    central_point = np.mean(camera_points, axis=0)
    distances = pairwise_distances(camera_points, [central_point])
    # print(distances)
    distance_99 = np.percentile(distances, 99)
    distance = distance_99 * 1.0
    return [central_point, distance]


def create_a_cube(camera_points):
    # Find the minimum and maximum coordinates of the camera points
    min_coords = np.min(camera_points, axis=0)
    max_coords = np.max(camera_points, axis=0)

    # Calculate the side length of the cube
    factor = 0.99
    side_length = max(max_coords - min_coords) * factor

    # Calculate the central point of the cube
    central_point = (min_coords + max_coords) / 2

    return [central_point, side_length]


def crop_a_point_cloud(point_cloud_data_file, parameters: list, shape=['sphere', 'cube']):
    vertices = np.asarray(point_cloud_data_file.points)
    colors = np.asarray(point_cloud_data_file.colors)

    if shape == 'sphere':
        # Extract the sphere parameters
        sphere_center, sphere_radius = parameters

        # Calculate the squared radius for efficient comparison
        sphere_radius_squared = sphere_radius ** 2

        # Calculate the squared distance from each point to the sphere center
        distances_squared = np.sum((vertices - sphere_center) ** 2, axis=1)

        # Find the indices of points inside the sphere
        indices_inside_sphere = np.where(distances_squared <= sphere_radius_squared)[0]

        # Extract the points and colors inside the sphere
        new_vertices = vertices[indices_inside_sphere]
        new_colors = colors[indices_inside_sphere]
    elif shape == 'cube':
        # Extract the cube parameters
        cube_center, side_length = parameters

        # Define the minimum and maximum coordinates of the cube
        min_coords = cube_center - side_length / 2
        max_coords = cube_center + side_length / 2

        # Find the indices of points inside the cube
        indices_inside_cube = np.all((vertices >= min_coords) & (vertices <= max_coords), axis=1)

        # Extract the points and colors inside the cube
        new_vertices = vertices[indices_inside_cube]
        new_colors = colors[indices_inside_cube]

    return new_vertices, new_colors


def calculate_rotation_and_scaling(circle_point_cloud):
    def fit_plane_least_squares(points):
        """
        Fit a plane to 3D points using least squares.

        Parameters:
            points (numpy array): 2D array of points with shape (n, 3).

        Returns:
            normal_vector (numpy array): Normal vector of the fitted plane.
        """

        # Define the objective function for least squares plane fitting
        def objective_function(params):
            a, b, c, d = params
            return np.sum((a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) ** 2)

        # Initial guess for the plane parameters
        initial_guess = np.ones(4)

        # Minimize the objective function to find the plane parameters
        result = minimize(objective_function, initial_guess, method='trust-constr')

        # Extract the normal vector of the fitted plane
        normal_vector = result.x[:3] / np.linalg.norm(result.x[:3])

        return normal_vector

    def fit_circle_least_squares(points):
        """
        Fit a circle to 2D points using least squares.

        Parameters:
            points (numpy array): 2D array of points with shape (n, 2).

        Returns:
            circle_center (numpy array): Center coordinates of the fitted circle.
            circle_radius (float): Radius of the fitted circle.
        """
        x, y = points[:, 0], points[:, 1]

        # Initial guess for the circle parameters
        initial_guess = np.mean(x), np.mean(y), np.mean(np.sqrt((x - np.mean(x)) ** 2 + (y - np.mean(y)) ** 2))

        # Define the objective function for least squares fitting
        objective_function = lambda params: np.sum(
            (np.sqrt((x - params[0]) ** 2 + (y - params[1]) ** 2) - params[2]) ** 2)

        # Minimize the objective function to find the circle parameters
        result = minimize(objective_function, initial_guess, method='trust-constr')

        # Extract the fitted circle parameters
        circle_center = result.x[:2]
        circle_radius = result.x[2]

        return circle_center, circle_radius

    # Convert input to NumPy arrays
    points = np.array(circle_point_cloud.points)

    # Fit a plane to the points in the circle point cloud
    normal_vector = fit_plane_least_squares(points)

    # Get the rotation matrix to align the plane with the XY plane
    z_axis = np.array([0, 0, 1])
    rotation_vector = np.cross(normal_vector, z_axis)
    rotation_angle = np.arccos(np.dot(normal_vector, z_axis))
    rotation_matrix = Rotation.from_rotvec(rotation_vector * rotation_angle).as_matrix()

    # Rotate the entire circle point cloud
    rotated_points = np.dot(points, rotation_matrix.T)

    # Project the rotated points onto the XY plane
    projected_points = rotated_points[:, :2]

    # Fit a circle to the 2D points on the XY plane
    fitted_circle_center, fitted_circle_radius = fit_circle_least_squares(projected_points)

    # Calculate the scaling factor based on the diameter of the fitted circle
    scale_factor = 15 / (2 * fitted_circle_radius)

    scaling_parameters = [rotation_matrix, scale_factor]

    return scaling_parameters


def transform_point_cloud(point_cloud, scaling_parameters):
    # Extract
    rotation_matrix, scale_factor = scaling_parameters
    # Convert input to NumPy arrays
    points = np.array(point_cloud.points)

    # Translate the point cloud so that the circle center is at the origin
    translated_point_cloud = points - np.mean(points, axis=0)

    # Apply rotation to the translated point cloud
    rotated_point_cloud = np.dot(translated_point_cloud, rotation_matrix.T)

    # Apply scaling to the rotated point cloud
    scaled_point_cloud = rotated_point_cloud * scale_factor

    # Extract original colors
    colors = np.array(point_cloud.colors)

    return scaled_point_cloud, colors


def green_index(point_cloud_data_file):
    vertices = np.asarray(point_cloud_data_file.points)
    colors = np.asarray(point_cloud_data_file.colors)
    print(colors.shape)

    green_indices = []
    # Find indices of the dots with Green Index > 0.8
    for i in range(len(colors)):
        green_index_value = (colors[i, 1] - colors[i, 0]) / (colors[i, 1] + colors[i, 0])
        if green_index_value > -0.02:
            green_indices.append(i)
    filtered_points = vertices[green_indices]
    filtered_colors = colors[green_indices]
    return filtered_points, filtered_colors


def red_index(point_cloud_data_file):
    vertices = np.asarray(point_cloud_data_file.points)
    colors = np.asarray(point_cloud_data_file.colors)
    print(colors.shape)

    red_indices = []
    # Find indices of the dots with Green Index > 0.8
    for i in range(len(colors)):
        red_index_value = (colors[i, 0] - colors[i, 1]) / (colors[i, 1] + colors[i, 0])
        if red_index_value > 0.25:
            red_indices.append(i)
    filtered_points = vertices[red_indices]
    filtered_colors = colors[red_indices]
    return filtered_points, filtered_colors


def create_alpha_shape(point_cloud_file_path, alpha, output_file_path=None):
    # Load point cloud from .ply file
    point_cloud = o3d.io.read_point_cloud(point_cloud_file_path)
    # Estimate normals
    point_cloud.estimate_normals()
    point_cloud.orient_normals_consistent_tangent_plane(100)
    # Compute alpha shape
    alpha_shape = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        point_cloud, alpha=alpha
    )
    # It allows to avoid saving a .ply with an alpha shape in case you don't need it
    if output_file_path is not None:
        # Save alpha shape to a .ply file
        o3d.io.write_triangle_mesh(output_file_path, alpha_shape)
    return alpha_shape


def create_poisson_shape(point_cloud_file_path, depth, output_file_path=None):
    # Load point cloud from .ply file
    point_cloud = o3d.io.read_point_cloud(point_cloud_file_path)
    # Estimate normals
    point_cloud.estimate_normals()
    point_cloud.orient_normals_consistent_tangent_plane(100)
    # Compute Poisson shape
    poisson_shape, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud,
        depth=depth
    )

    # It allows to avoid saving a .ply with an alpha shape in case you don't need it
    if output_file_path:
        # Save Poisson shape to a .ply file
        o3d.io.write_triangle_mesh(output_file_path, poisson_shape)

    # ChatGPT tells me that I can make sure everuthing is watertight right away. I haven't cheked it yet. If it's not working, I can use the calculate_watertight_volume function.
    # Ensure mesh is watertight
    poisson_shape.compute_vertex_normals()
    poisson_shape.remove_degenerate_triangles()
    poisson_shape.remove_unreferenced_vertices()

    return poisson_shape


def create_ball_pivoting_shape(point_cloud_file_path, radii, output_file_path=None):
    # Load point cloud from .ply file
    point_cloud = o3d.io.read_point_cloud(point_cloud_file_path)
    # Estimate normals
    point_cloud.estimate_normals()
    point_cloud.orient_normals_consistent_tangent_plane(100)
    # Compute Ball pivoting shape
    ball_pivoting_shape = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud, o3d.utility.DoubleVector(radii))

    # It allows to avoid saving a .ply with an alpha shape in case you don't need it
    if output_file_path:
        # Save Ball pivoting shape to a .ply file
        o3d.io.write_triangle_mesh(output_file_path, ball_pivoting_shape)

    # Ensure mesh is watertight
    ball_pivoting_shape.compute_vertex_normals()
    ball_pivoting_shape.remove_degenerate_triangles()
    ball_pivoting_shape.remove_unreferenced_vertices()

    return ball_pivoting_shape


def create_marching_cubes_shape(point_cloud_file_path, threshold, output_file_path=None, target_resolution=1.0):
    # Load point cloud from .ply file
    point_cloud = pv.read(point_cloud_file_path)

    # Convert the point cloud to NumPy array
    points = np.array(point_cloud.points)

    # Calculate the bounding box of the point cloud
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)

    # Calculate the dimensions of the bounding box
    dimensions = max_coords - min_coords

    # Calculate the number of bins based on target resolution
    num_bins = np.ceil(np.max(dimensions) / target_resolution).astype(int)

    # Create a 3D scalar field from the point cloud
    values, edges = np.histogramdd(points, bins=num_bins, range=[(min_coords[0], max_coords[0]),
                                                                 (min_coords[1], max_coords[1]),
                                                                 (min_coords[2], max_coords[2])])

    # Threshold the scalar field to extract the surface
    vertices, triangles = mcubes.marching_cubes(values, threshold)

    # Scale vertices to match the original point cloud
    vertices *= (np.max(dimensions) / (num_bins - 1))
    vertices += min_coords

    # Create an o3d mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # It allows to avoid saving a .ply in case you don't need it
    if output_file_path:
        # Save Marching Cubes shape to a .ply file
        o3d.io.write_triangle_mesh(output_file_path, mesh)

    return mesh


def create_convex_hull_shape(point_cloud_file_path, output_file_path=None):
    # Load point cloud from .ply file
    point_cloud = o3d.io.read_point_cloud(point_cloud_file_path)
    # Estimate normals
    point_cloud.estimate_normals()
    point_cloud.orient_normals_consistent_tangent_plane(100)
    # Compute convex_hull shape
    convex_hull_shape, _ = point_cloud.compute_convex_hull()

    # It allows to avoid saving a .ply with an alpha shape in case you don't need it
    if output_file_path:
        # Save convex_hull shape to a .ply file
        o3d.io.write_triangle_mesh(output_file_path, convex_hull_shape)

    return convex_hull_shape


def calculate_watertight_volume(shape):
    vertices = np.asarray(shape.vertices)
    triangles = np.asarray(shape.triangles)
    # Create a Trimesh object
    # This library allows to make sure that each mesh is watertight
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    total_volume = 0
    for i in mesh.split():
        if i.is_watertight:
            # print(i.volume)
            total_volume += abs(i.volume)
    print('Total volume of watertight triangles: {}'.format(total_volume))
    return total_volume


def calculate_shape_parameters(point_cloud_file_path, mesh, total_volume):
    shape = o3d.io.read_triangle_mesh(mesh)
    point_cloud = o3d.io.read_point_cloud(point_cloud_file_path)
    point_cloud_array = np.asarray(point_cloud.points)
    print('files loaded')
    # Get dimensions (length, width, height)
    dimensions = np.ptp(point_cloud_array, axis=0)
    print('dimensions calculated')
    # Swap dimensions for height, length, and width
    dimensions = dimensions[[1, 2, 0]]
    # Calculate bounding box
    # Compute the axis-aligned bounding box (AABB)
    bounding_box = shape.get_axis_aligned_bounding_box()

    # Extract the minimum and maximum points of the bounding box
    min_point = bounding_box.min_bound
    max_point = bounding_box.max_bound

    # Vertices of the bounding box in the XY plane
    vertices = np.array([
        [min_point[0], min_point[1]],
        [min_point[0], max_point[1]],
        [max_point[0], min_point[1]],
        [max_point[0], max_point[1]]
    ])

    # Calculate all pairwise distances in the XY plane
    pairwise_distances = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            distance = np.linalg.norm(vertices[i] - vertices[j])
            pairwise_distances.append(distance)

    # Identify the shortest and longest distances
    width = min(pairwise_distances)
    length = max(pairwise_distances)

    # Calculate the height
    height = max_point[2] - min_point[2]

    # Calculate volumes and areas
    bounding_box_volume = length * width * height
    bounding_box_xy_area = width * length

    print('bounding boxes calculated')
    # Calculate surface area
    surface_area = shape.get_surface_area()
    print('surface area calculated')
    # Calculate aspect ratio
    try:
        aspect_ratio = np.max(dimensions) / np.min(dimensions)
    except ZeroDivisionError:
        print("Can't calculate aspect ratio. Division by zero. Aspect Ratio set to 0")
        aspect_ratio = 0
    print('aspect ratio calculated')
    # # Calculate elongation
    # try:
    #     elongation = (np.max(dimensions) / np.median(dimensions)) - 1
    # except ZeroDivisionError:
    #     print("Can't calculate elongation. Division by zero.")
    #     elongation = 0
    # print('elongation calculated')
    # # Calculate flatness
    # try:
    #     flatness = (np.min(dimensions) / np.median(dimensions)) - 1
    # except ZeroDivisionError:
    #     print("Can't calculate flatness. Division by zero.")
    #     flatness = 0
    # print('flatness calculated')
    # Get connected components
    # connected_components = shape.cluster_connected_triangles()
    # print('connected components calculated')
    # Initialize a dictionary to store parameters for each component
    component_parameters = {}

    # Calculate sphericity for the entire alpha shape
    # try:
    #     sphericity = (np.pi ** (1 / 3)) * ((6 * total_volume) ** (2 / 3)) / surface_area
    # except ZeroDivisionError:
    #     print("Can't calculate sphericity. Division by zero.")
    #     sphericity = 0
    # print('sphericity calculated')
    # # Calculate compactness for the entire alpha shape
    # try:
    #     compactness = (36 * np.pi * total_volume ** 2) ** (1 / 3) / surface_area
    # except ZeroDivisionError:
    #     print("Can't calculate compactness. Division by zero.")
    #     compactness = 0
    # print('compactness calculated')
    # Calculate number of independent components
    vertices = np.asarray(shape.vertices)
    triangles = np.asarray(shape.triangles)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    components = 0
    for i in mesh.split():
        components += 1
    print('components calculated')
    # # Get points inside the alpha shape
    # points_inside = shape.select_by_index(np.arange(len(point_cloud)))

    # Get the number of points inside the alpha shape
    num_points_inside = len(point_cloud_array)
    try:
        point_density = num_points_inside / total_volume
    except ZeroDivisionError:
        print("Can't calculate point density. Division by zero.")
        point_density = 0
    print('point density calculated')
    # Store parameters for the entire alpha shape
    parameters = {
        'height': height,
        'length': length,
        'width': width,
        'volume': total_volume,
        'surface_area': surface_area,
        'aspect_ratio': aspect_ratio,
        'bounding_box_volume': bounding_box_volume,
        'bounding_box_xy_area': bounding_box_xy_area,
        'components_number': components
    }
    return parameters


def calculate_statistics_trimesh(mesh_file_path, repair=True):
    """
    :param mesh_file_path: Path to your .ply mesh file
    :return: A list of two dictionaries containing all the statistics of the mesh without any corrections
    :repair: Bool to set if you want to repair the mesh
    """

    def calculate_parameters(mesh):
        try:
            # Initiate a dictionary
            results = {}

            # 1. Get dimensions:
            # Calculate the axis-aligned bounding box (AABB)
            bounding_box = mesh.bounds
            # Extract the minimum and maximum points of the bounding box
            min_point = bounding_box[0]
            max_point = bounding_box[1]

            # Vertices of the bounding box
            vertices = np.array([
                [min_point[0], min_point[1]],
                [min_point[0], max_point[1]],
                [max_point[0], min_point[1]],
                [max_point[0], max_point[1]]
            ])

            # Calculate all pairwise distances in the XY plane
            pairwise_distances = []
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    distance = np.linalg.norm(vertices[i] - vertices[j])
                    pairwise_distances.append(distance)

            # Identify the shortest and longest distances
            width = min(pairwise_distances)
            length = max(pairwise_distances)

            # Calculate the height
            height = max_point[2] - min_point[2]

            # Calculate volumes and areas
            bounding_box_volume = length * width * height
            bounding_box_xy_area = width * length

            # 2. Get volume
            # Calculate using trimesh
            vol = mesh.volume

            # 3. Get surface area
            # Calculate using trimesh
            surface_area = mesh.area

            # 4. Get components number
            # Calculate by splitting the mesh into pieces
            components = len(mesh.split(only_watertight=False))

            # 5. Get aspect ratio
            # Calculate it using previous data
            aspect_ratio = length / width

            # Print out
            print(f"Height: {height}")
            print(f"Width: {width}")
            print(f"Length: {length}")
            print(f"Bounding box volume: {bounding_box_volume}")
            print(f"Bounding box xy area: {bounding_box_xy_area}")
            print(f"Volume: {vol}")
            print(f"Surface area: {surface_area}")
            print(f"Aspect ratio: {aspect_ratio}")
            print(f"Components: {components}")

            # Create dictionary
            results['height'] = height
            results['width'] = width
            results['length'] = length
            results['volume'] = vol
            results['surface_area'] = surface_area
            results['aspect_ratio'] = aspect_ratio
            results['bounding_box_volume'] = bounding_box_volume
            results['bounding_box_xy_area'] = bounding_box_xy_area
            results['components_number'] = components

            return results

        except ValueError:
            print("Value Error. Something is wrong with the mesh")
        except ZeroDivisionError:
            print("Can't calculate statistics. Division by zero.")
        except Exception as e:
            print(f"Unexpected error: {e}")

    if repair:
        # Load a mesh
        mesh = trimesh.load(mesh_file_path)
        print(mesh_file_path)
        print('Stats for the mesh: ')
        # Calculate first statistics for the simples
        simples = calculate_parameters(mesh)

        # Fix the mesh in steps to avoid high memory usage
        try:
            mesh.merge_vertices,
            trimesh.repair.fix_inversion,
            trimesh.repair.fix_normals,
            trimesh.repair.fill_holes,
            trimesh.repair.fix_winding,
            trimesh.repair.fill_holes,
            trimesh.repair.fix_normals,
            print('Stats for the repaired mesh: ')
            # Calculate statistics for the repaired mesh
            repaired = calculate_parameters(mesh)
        except Exception as e:
            print('Error')
            repaired = {'repaired': 'repaired'}


        return [simples, repaired]
    else:
        # Load a mesh
        mesh = trimesh.load(mesh_file_path)
        print(mesh_file_path)
        print('Stats for the mesh: ')
        # Calculate first statistics for the simples
        simples = calculate_parameters(mesh)

        return simples


def eliminate_outliers_and_scale(file_path, export_path):
    df = pd.read_csv(file_path)
    print(f"Processing file: {file_path}")
    cols = [
        'height', 'length', 'width', 'volume', 'surface_area', 'aspect_ratio',
        'components_number', 'measured_leaf_area'
    ]

    # Remove rows where any of the specified columns are zero
    for column in cols:
        df = df[df[column] != 0]

    # Calculate limits for outlier removal
    limits = {}
    for column in cols:
        data = df[column]
        z_limit_top = 3 * data.std() + data.mean()
        z_limit_bottom = -3 * data.std() + data.mean()
        limits[column] = [z_limit_top, z_limit_bottom]

    # Print limits for debugging
    for column, limit in limits.items():
        print(f"{column} limits: {limit[1]}, {limit[0]}")

    # Remove outliers
    for column, limit in limits.items():
        df = df[(df[column] < limit[0]) & (df[column] > limit[1])]
    df.reset_index(drop=True, inplace=True)

    # Initialize scaler
    scaler = None

    # Check if there are any rows left after outlier removal
    if len(df) > 0:
        print(f'Number of rows after outlier removal: {len(df)}')

        # Print min and max of each column for debugging
        for column in df.columns:
            print(f"{column} min: {df[column].min()}, max: {df[column].max()}")

        # Create an instance of StandardScaler
        scaler = StandardScaler()

        # Fit the scaler on the data and transform it, excluding 'experiment_number'
        scaling_columns = df.columns.difference(['experiment_number'])
        scaled_data = scaler.fit_transform(df[scaling_columns])

        # Convert the scaled data back to a DataFrame
        scaled_df = pd.DataFrame(scaled_data, columns=scaling_columns)
        scaled_df['experiment_number'] = df['experiment_number'].values  # Add 'experiment_number' back to the DataFrame

        # Save the processed file
        scaled_df.to_csv(export_path, index=False)
        print(f"Processed file saved to: {export_path}")
    else:
        print("No data left after outlier removal. Exporting an empty DataFrame.")
        pd.DataFrame().to_csv(export_path, index=False)
    return scaler

if __name__ == '__main__':
    folders_paths = paths.get_paths()

    # Set the runtype parameter
    runtype_options = [
        'whole',                # All files up to scaling not included
        'one_file',             # Same procedure for one file
        'red_from_clipped',     # Get red_from_clipped from clipped/cubes folder
        'scale',                # Run procedure of scaling using circles
        'create_alpha',         # Create alpha shapes and the csv based on corrected folder, get stats
    ]
    runtype = 'test'

    if runtype == 'test':
        for x, y in folders_paths.items():
            print(x, y)
        for file in os.listdir(folders_paths['plys']):
            ply_file_path = os.path.join(folders_paths['plys'], file)
            if os.path.isfile(ply_file_path) and ply_file_path.lower().endswith('.ply'):
                print(file)
                open_ply_file(os.path.join(folders_paths['plys'], file))
else:
    print('loaded')
