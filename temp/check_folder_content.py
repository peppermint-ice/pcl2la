import os
from config import paths
import pandas as pd

df = pd.DataFrame(columns=['index', 'filename'])
folder_path = paths.get_paths()['ready_for_training']

for i, file in enumerate(os.listdir(folder_path)):
    if file.endswith('noElim.csv'):
        df = pd.concat([df, pd.DataFrame({'index': [i], 'filename': [file]})], ignore_index=True)
# df.to_csv('contents.csv', index=False)

for file in os.listdir(folder_path):
    if file.endswith('o3d_simple_noElim.csv'):
        print(file)
for file in os.listdir(folder_path):
    if file.endswith('trimesh_simple_noElim.csv'):
        print(file)
for file in os.listdir(folder_path):
    if file.endswith('trimesh_repaired_noElim.csv'):
        print(file)