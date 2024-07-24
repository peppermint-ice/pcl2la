import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import paths

# Set paths
folder_paths = paths.get_paths()
combined_folder_path = folder_paths["combined"]
plots_all_reconstructions_folder_path = folder_paths["plots_all_reconstructions"]
file_name = "combine_ML_240724_1608_filtered.csv"
file_path = os.path.join(combined_folder_path, file_name)

# Load the filtered dataset
df_filtered = pd.read_csv(file_path)

# Replace underscores with spaces and capitalize labels
df_filtered['algorithm_name'] = df_filtered['algorithm_name'].str.replace('_', ' ').str.capitalize()
df_filtered['regression_model'] = df_filtered['regression_model'].replace({
    'rf': 'Random Forest',
    'xgb': 'XGBoost',
    'en': 'Elastic Net',
    'lasso': 'Lasso',
    'ridge': 'Ridge',
    'mlp': 'MLP',
    'linear': 'Linear'
})

# Define colors for each regression model
colors = {
    'XGBoost': 'blue',
    'Random Forest': 'green',
    'Elastic Net': 'orange',
    'Lasso': 'purple',
    'Ridge': 'red',
    'MLP': 'brown',
    'Linear': 'pink'
}

# 1. Plot all R2s
# Get unique algorithm names
algorithm_names = df_filtered['algorithm_name'].unique()

# Create independent plots for each algorithm
for algorithm in algorithm_names:
    subset = df_filtered[df_filtered['algorithm_name'] == algorithm]
    plt.figure(figsize=(12, 5))
    plt.grid(True, which='both', axis='y', zorder=1)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    ax = sns.barplot(
        x='parameter_value', y='R2_global_test', hue='regression_model', data=subset, palette=colors, zorder=2
    )
    plt.title(f'R2 on test set for {algorithm}')
    plt.xlabel('Parameter Value')
    plt.ylabel('R2 on test subset')
    plt.legend(title='Regression Model')

    # Save the plot with abstract names
    plot_filename = f"R2_test_{algorithm.replace(' ', '_')}.png"
    plot_filepath = os.path.join(plots_all_reconstructions_folder_path, plot_filename)
    plt.savefig(plot_filepath)
    plt.close()

# 2. Plot best R2 for each algorithm

# Identify the best parameter value for each algorithm based on R2_global_test
best_params = df_filtered.loc[df_filtered.groupby(['algorithm_name', 'regression_model'])['R2_global_test'].idxmax()]

# Plot 1: Comparison of Regression Models Based on Best Parameter Values
plt.figure(figsize=(14, 8))
ax = sns.barplot(x='algorithm_name', y='R2_global_test', hue='regression_model', data=best_params, palette=colors, zorder=3)
plt.title('Comparison of Regression Models Based on Best Parameter Values')
plt.xlabel('Algorithm Name')
plt.ylabel('R2 on test subset')
plt.legend(title='Regression Model')
plt.grid(axis='y', zorder=0)

# Add R² values on the bars
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=10, color='black', zorder=4)

# Save the plot with an abstract name
plot_filename = "Best_R2_per_Algorithm.png"
plot_filepath = os.path.join(plots_all_reconstructions_folder_path, plot_filename)
plt.savefig(plot_filepath)
plt.close()

# Change axes to compare reconstruction methods
pivot_data = best_params.pivot(index='regression_model', columns='algorithm_name', values='R2_global_test')

# Plot 2: Comparison of Reconstruction Methods
plt.figure(figsize=(14, 8))
ax = pivot_data.plot(kind='bar', figsize=(14, 8), zorder=3, color=[colors.get(x, '#333333') for x in pivot_data.columns])
plt.title('Comparison of Reconstruction Methods')
plt.xlabel('Regression Model')
plt.ylabel('R2 on test subset')
plt.grid(axis='y', zorder=0)

# Add R² values on the bars
for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', (bar.get_x() + bar.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, color='black', zorder=4)

# Save the plot with an abstract name
plot_filename = "Comparison_Reconstruction_Methods.png"
plot_filepath = os.path.join(plots_all_reconstructions_folder_path, plot_filename)
plt.savefig(plot_filepath)
plt.close()
