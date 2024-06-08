import re

# Sample filenames
filenames = [
    "0.005_alpha_o3d_repaired_noElim.csv",
    "12_poisson_trimesh_simple_elim.csv"
]

# Regular expression pattern
pattern = r'^([0-9.]+)_([a-zA-Z0-9_]+)_([a-zA-Z0-9_]+)_([a-zA-Z0-9_]+)_(.*)\.csv$'

# Extract and print the parts for each filename
for filename in filenames:
    match = re.match(pattern, filename)
    if match:
        print(f"Filename: {filename}")
        print(match.groups()[0])
        print(match.groups()[1])
        print(match.groups()[2])
        print(match.groups()[3])
        print(match.groups()[4])