import os
from config import paths
import pandas as pd
import re


df = pd.DataFrame(columns=['filename', 'exp', 'plant', 'cut', 'leaf_area'])
folder_path = "G:\My Drive\Dmitrii - Ph.D Thesis\Frost room Experiment Data\LA\plys\corrected_ready_to_execute"
export_folder_path = paths.get_paths()['combined']

for i, file in enumerate(os.listdir(folder_path)):
    if file.endswith('.ply'):
        pattern = r'(\d*)_(\d*)_(\w*)_(\d*p?\d*)\.ply'
        match = re.search(pattern, file)
        if match:
            exp = match.group(1)
            if exp[4:] == '23':
                exp = 1
            elif exp[4:] == '24':
                exp = 2
            plant = match.group(2)
            cut = match.group(3)
            leaf_area = match.group(4)
            if "_" in cut:
                cut = cut.split("_")[0]
            if int(plant) <= 15:
                group = 'contol'
            elif int(plant) <= 30:
                group = 't1'
            elif int(plant) <= 45:
                group = 't2'
            elif int(plant) <= 60:
                group = 't3'
            if int(plant) in [1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 31, 32, 33, 34, 35, 46, 47, 48, 49, 50]:
                cultivar = 'Mohammed'
            elif int(plant) in [6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 36, 37, 38, 39, 40, 51, 52, 53, 54, 55]:
                cultivar = 'Hahms Gelbe'
            elif int(plant) in [11, 12, 13, 14, 15, 26, 27, 28,29, 30, 41, 42, 43, 44, 45, 56, 57, 58, 59, 60]:
                cultivar = 'Red Robin'
        df = pd.concat([df, pd.DataFrame({'exp': [int(exp)], 'plant': [int(plant)], 'cut': [int(cut)], 'cultivar': [cultivar], 'treatment': [group], 'leaf_area': [float(leaf_area.replace("p", "."))]})], ignore_index=True)
df.to_csv(os.path.join(export_folder_path, 'all_plants.csv'), index=False)
df.sort_values(by=['exp'], inplace=True)
print(df.to_string())

df_1 = df[df['exp'] == 1]
df_2 = df[df['exp'] == 2]
print(len(df_1['plant'].unique()))
print(len(df_1))
print(len(df_2['plant'].unique()))
print(len(df_2))


# for file in os.listdir(folder_path):
#     if file.endswith('o3d_simple_noElim.csv'):
#         print(file)
# for file in os.listdir(folder_path):
#     if file.endswith('trimesh_simple_noElim.csv'):
#         print(file)
# for file in os.listdir(folder_path):
#     if file.endswith('trimesh_repaired_noElim.csv'):
#         print(file)