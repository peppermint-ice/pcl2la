import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def extract_year_from_filename(filename):
    pattern = r'^(\d{6})_(\d+)_\d+_\d+p\d+\.ply$'
    match = re.match(pattern, filename)
    if match:
        date_str = match.group(1)
        year = 2023 if date_str.endswith('23') else 2024
        return year
    return None


def load_data_with_year(directory):
    data = []

    for filename in os.listdir(directory):
        if filename.endswith('.ply'):
            year = extract_year_from_filename(filename)
            if year is not None:
                # Extract measured leaf area
                match = re.search(r'(\d+p\d+)\.ply$', filename)
                if match:
                    measured_leaf_area = float(match.group(1).replace('p', '.'))
                    data.append({'Year': year, 'Measured_Leaf_Area': measured_leaf_area})

    return pd.DataFrame(data)


if __name__ == '__main__':
    # Define the specific directory path
    directory_path = r'C:\Users\dusen\Documents\PhD\plys\meshes\marching_cubes\0.8'

    # Load the data with year and measured leaf area
    df = load_data_with_year(directory_path)

    # Plot distributions
    plt.figure(figsize=(10, 6))
    plot = sns.histplot(df, x='Measured_Leaf_Area', hue='Year', kde=True, element='step', palette='tab10',
                        stat='density')

    plt.title('Distribution of Measured Leaf Area by Experiment Year')
    plt.xlabel('Measured Leaf Area')
    plt.ylabel('Density')

    # Correct the legend
    handles, labels = plot.get_legend_handles_labels()
    labels = ['2023', '2024']  # Replace labels with the correct years
    plt.legend(handles, labels, title='Year')

    plt.show()
