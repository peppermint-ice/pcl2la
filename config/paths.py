import json
import os


def get_paths():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    with open(os.path.join(PROJECT_ROOT, 'config', 'config.json')) as f:
        config = json.load(f)
    with open(os.path.join(PROJECT_ROOT, 'config', 'current_pc.json')) as f:
        current_pc = json.load(f)
    hostname = current_pc['pc_name']

    paths = {
        "plys": config[hostname]["plys"],
        "clipped_sph": os.path.join(config[hostname]["plys"], "clipped/spheres"),
        "red_from_clipped_sph": os.path.join(config[hostname]["plys"], "clipped/spheres/red_from_clipped"),
        "clustered_sph": os.path.join(config[hostname]["plys"], "clipped/spheres/clustered"),
        "green_sph": os.path.join(config[hostname]["plys"], "clipped/spheres/clustered/color_filtered/green"),
        "rotated_sph": os.path.join(config[hostname]["plys"], "clipped/spheres/clustered/color_filtered/green/rotated"),
        "red_sph": os.path.join(config[hostname]["plys"], "clipped/spheres/clustered/color_filtered/red"),
        "clipped_cubes": os.path.join(config[hostname]["plys"], "clipped/cubes"),
        "red_from_clipped_cubes": os.path.join(config[hostname]["plys"], "clipped/cubes/red_from_clipped"),
        "clustered_cubes": os.path.join(config[hostname]["plys"], "clipped/cubes/clustered"),
        "green_cubes": os.path.join(config[hostname]["plys"], "clipped/cubes/clustered/color_filtered/green"),
        "rotated_cubes": os.path.join(config[hostname]["plys"], "clipped/cubes/clustered/color_filtered/green/rotated"),
        "red_cubes": os.path.join(config[hostname]["plys"], "clipped/cubes/clustered/color_filtered/red"),
        "meshes": os.path.join(config[hostname]["plys"], "meshes"),
        "circles": os.path.join(config[hostname]["plys"], "circles"),
        "corrected": os.path.join(config[hostname]["plys"], "corrected_ready_to_execute"),
        "alphas": os.path.join(config[hostname]["plys"], "meshes/alpha"),
        "poissons": os.path.join(config[hostname]["plys"], "meshes/poisson"),
        "marching_cubes": os.path.join(config[hostname]["plys"], "meshes/marching_cubes"),
        "bvas": os.path.join(config[hostname]["plys"], "meshes/bva"),
        "ball_pivoting": os.path.join(config[hostname]["plys"], "meshes/ball_pivoting"),
        "convex_hull": os.path.join(config[hostname]["plys"], "meshes/convex_hull"),
        "data": os.path.join(config[hostname]["plys"], "data"),
        "results": os.path.join(config[hostname]["plys"], "data/results"),
        "combined": os.path.join(config[hostname]["plys"], "data/combined"),
        "ready_for_training": os.path.join(config[hostname]["plys"], "data/ready_for_training")
    }

    return paths
