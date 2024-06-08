import pandas as pd
import os

from config import paths

if __name__ == '__main__':
    folder_paths = paths.get_paths()
    csv_folder_path = folder_paths["data"]
    import_csv_path = os.path.join(csv_folder_path, "005final.csv")
    export_csv_path = os.path.join(csv_folder_path, "006final_no_outliers.csv")

    df = pd.read_csv(import_csv_path)
    cols = [
        'Height', 'Length', 'Width', 'Volume', 'Surface_area', 'Aspect_ratio',
        'Components_number', 'Point_density', 'Measured_leaf_area'
    ]

    for column in cols:
        df = df[df[column] != 0]

    limits = {}
    for column in cols:
        data = df[column]
        z_limit_top = 3 * data.std() + data.mean()
        z_limit_bottom = -3 * data.std() + data.mean()
        limits[column] = [z_limit_top, z_limit_bottom]

    for column, limit in limits.items():
        print(column, limit[1], limit[0])

    for column, limit in limits.items():
        df = df[(df[column] < limit[0]) & (df[column] > limit[1])]
    df.reset_index(drop=True, inplace=True)

    print('after:', len(df))
    for column in df.columns:
        print(column, df[column].min(), df[column].max())
    df.to_csv(export_csv_path, index=False)