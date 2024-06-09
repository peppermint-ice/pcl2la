import re
import os
from config import paths

# # Sample filenames
# filenames = [
#     "alpha_0.01_o3d_repaired_noElim.csv",
#     "poisson_12_trimesh_simple_elim.csv",
#     "marching_cubes_12_trimesh_simple_elim.csv"
# ]
#
# # Regular expression pattern
# pattern = r'^([a-zA-Z0-9_]+)_([0-9.]+)_([a-zA-Z0-9_]+)_([a-zA-Z0-9_]+)_(.*)\.csv$'
#
# # Extract and print the parts for each filename
# for filename in filenames:
#     match = re.match(pattern, filename)
#     if match:
#         print(f"Filename: {filename}")
#         print(match.groups()[0])
#         print(match.groups()[1])
#         print(match.groups()[2])
#         print(match.groups()[3])
#         print(match.groups()[4])

folder_paths = paths.get_paths()
folder_path = paths["ready_for_training"]
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        # Construct the full file path
        old_file = os.path.join(folder_path, filename)

        # Split the filename and extension
        name, ext = os.path.splitext(filename)

        # Create the new filename
        new_filename = name + '_noElim' + ext
        new_file = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(old_file, new_file)