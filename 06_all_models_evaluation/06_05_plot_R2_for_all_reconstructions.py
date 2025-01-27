import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import paths


def change_column_names(df):
    # Replace underscores with spaces and capitalize labels
    df['algorithm_name'] = df['algorithm_name'].str.replace('_', ' ').str.capitalize()
    df['regression_model'] = df['regression_model'].replace({
        'rf': 'Random Forest',
        'xgb': 'XGBoost',
        'en': 'Elastic Net',
        'lasso': 'Lasso',
        'ridge': 'Ridge',
        'mlp': 'MLP',
        'linear': 'Linear'
    })
    return df

fig_size = (20, 6)

for year_param in ['noyear', 'byyear', 'byyearreveresed']:
    # Set paths
    folder_paths = paths.get_paths()
    combined_folder_path = folder_paths["combined"]
    plots_all_reconstructions_folder_path = folder_paths["plots_all_reconstructions"]

    #       R2
    # Load the file
    file_name = "combine_ML_060824_1807_filtered_r2.csv"
    file_path = os.path.join(combined_folder_path, file_name)

    print(year_param)

    # Load the filtered dataset
    df_filtered = pd.read_csv(file_path)
    df_filtered = df_filtered[df_filtered['byyear'] == year_param]

    # Replace underscores with spaces and capitalize labels. Get new column names
    df_filtered = change_column_names(df_filtered)

    # Define colors for each regression model and reconstruction method
    model_colors = {
        'XGBoost': 'blue',
        'Random Forest': 'green',
        'Elastic Net': 'orange',
        'Lasso': 'purple',
        'Ridge': 'red',
        'MLP': 'brown',
        'Linear': 'pink'
    }

    reconstruction_colors = {
        'Alpha': 'blue',
        'Marching cubes': 'green',
        'Ball pivoting': 'orange',
        'Poisson': 'purple'
    }

    # 1. Plot all R2s
    # Get unique algorithm names
    algorithm_names = df_filtered['algorithm_name'].unique()

    # Create independent plots for each algorithm
    # for algorithm in algorithm_names:
    #     subset = df_filtered[df_filtered['algorithm_name'] == algorithm]
    #     plt.figure(figsize=fig_size)
    #     plt.grid(True, which='both', axis='y', zorder=1)
    #     plt.grid(color='gray', linestyle='--', linewidth=0.5)
    #     ax = sns.barplot(
    #         x='parameter_value', y='R2_global_test', hue='regression_model', data=subset, palette=model_colors, zorder=2
    #     )
    #     if year_param == 'byyear':
    #         plt.title(f'R² on test set for {algorithm}. Predicting experiment #2 on data from experiment #1')
    #     elif year_param == 'byyearreveresed':
    #         plt.title(f'R² on test set for {algorithm}. Predicting experiment #1 on data from experiment #2')
    #     else:
    #         plt.title(f'R² on test set for {algorithm}')
    #     plt.xlabel('Parameter Value')
    #     plt.ylabel('R² on test subset')
    #     plt.legend(title='Regression Model', loc='lower right')
    #
    #     # Save the plot with abstract names
    #     plot_filename = f"{year_param}_R2_test_{algorithm.replace(' ', '_')}.png"
    #     plot_filepath = os.path.join(plots_all_reconstructions_folder_path, plot_filename)
    #     # plt.savefig(plot_filepath)
    #     # plt.close()
    #
    #     plt.show()

    # 2. Plot best R2 for each algorithm

    # Identify the best parameter value for each algorithm based on R2_global_test
    best_params = df_filtered.loc[df_filtered.groupby(['algorithm_name', 'regression_model'])['R2_global_test'].idxmax()]



    # Plot 1: Comparison of Regression Models Based on Best Parameter Values
    plt.figure(figsize=fig_size)
    ax = sns.barplot(x='algorithm_name', y='R2_global_test', hue='regression_model', data=best_params, palette=model_colors, zorder=3)
    if year_param == 'byyear':
        plt.title('Comparison of Regression Models Based on Best Parameter Values. Predicting experiment #2 on data from experiment #1')
    elif year_param == 'byyearreveresed':
        plt.title('Comparison of Regression Models Based on Best Parameter Values. Predicting experiment #1 on data from experiment #2')
    else:
        plt.title('Comparison of Regression Models Based on Best Parameter Values')
    plt.xlabel('Algorithm Name')
    plt.ylabel('R² on test subset')
    plt.legend(title='Regression Model', loc='lower right')
    plt.grid(axis='y', zorder=0)

    # Add R² values on the bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, color='black', zorder=4)

    # Save the plot with an abstract name
    plot_filename = f"{year_param}_R2_Best_per_Algorithm.png"
    plot_filepath = os.path.join(plots_all_reconstructions_folder_path, plot_filename)
    # plt.savefig(plot_filepath)
    # plt.close()
    plt.show()

    # Change axes to compare reconstruction methods
    pivot_data = best_params.pivot(index='regression_model', columns='algorithm_name', values='R2_global_test')

    # Plot 2: Comparison of Reconstruction Methods
    plt.figure(figsize=fig_size)
    ax = pivot_data.plot(kind='bar', figsize=fig_size, zorder=3)
    if year_param == 'byyear':
        plt.title('Comparison of Reconstruction Methods. Predicting experiment #2 on data from experiment #1')
    elif year_param == 'byyearreveresed':
        plt.title('Comparison of Reconstruction Methods. Predicting experiment #1 on data from experiment #2')
    else:
        plt.title('Comparison of Reconstruction Methods')
    plt.xlabel('Regression Model')
    plt.ylabel('R² on test subset')
    plt.grid(axis='y', zorder=0)

    # Add R² values on the bars
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', (bar.get_x() + bar.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, color='black', zorder=4)

    plt.legend(title='Algorithm Name', loc='lower right')
    plt.xticks(rotation=0)  # Set x-axis labels to be horizontal

    # Save the plot with an abstract name
    plot_filename = f"{year_param}_R2_Comparison_Reconstruction_Methods"
    plot_filepath = os.path.join(plots_all_reconstructions_folder_path, plot_filename)
    # plt.savefig(plot_filepath)
    # plt.close()
    plt.show()

    #       RMSE

    # Load the file
    file_name = "2507data_filtered_rmse.csv"
    file_path = os.path.join(combined_folder_path, file_name)

    # Load the filtered dataset
    df_filtered = pd.read_csv(file_path)
    df_filtered = df_filtered[df_filtered['byyear'] == year_param]

    # Replace underscores with spaces and capitalize labels. Get new column names
    df_filtered = change_column_names(df_filtered)

    # 1. Plot all RMSEs
    # Get unique algorithm names
    algorithm_names = df_filtered['algorithm_name'].unique()

    # Create independent plots for each algorithm
    for algorithm in algorithm_names:
        subset = df_filtered[df_filtered['algorithm_name'] == algorithm]
        plt.figure(figsize=fig_size)
        plt.grid(True, which='both', axis='y', zorder=1)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        ax = sns.barplot(
            x='parameter_value', y='RMSE_global_test', hue='regression_model', data=subset, palette=model_colors, zorder=2
        )
        if year_param == 'byyear':
            plt.title(f'RMSE on test set for {algorithm}. Predicting experiment #2 on data from experiment #1')
        elif year_param == 'byyearreveresed':
            plt.title(f'RMSE on test set for {algorithm}. Predicting experiment #1 on data from experiment #2')
        else:
            plt.title(f'RMSE on test set for {algorithm}')
        plt.xlabel('Parameter Value')
        plt.ylabel('RMSE on test subset, cm²')
        plt.legend(title='Regression Model', loc='lower right')

        # Save the plot with abstract names
        plot_filename = f"{year_param}_RMSE_test_{algorithm.replace(' ', '_')}.png"
        plot_filepath = os.path.join(plots_all_reconstructions_folder_path, plot_filename)
        plt.savefig(plot_filepath)
        plt.close()

    # # 2. Plot best RMSE for each algorithm
    #
    # # Identify the best parameter value for each algorithm based on R2_global_test
    # best_params = df_filtered.loc[df_filtered.groupby(['algorithm_name', 'regression_model'])['RMSE_global_test'].idxmin()]
    #
    # # Plot 1: Comparison of Regression Models Based on Best Parameter Values
    # plt.figure(figsize=fig_size)
    # ax = sns.barplot(x='algorithm_name', y='RMSE_global_test', hue='regression_model', data=best_params, palette=model_colors, zorder=3)
    # if year_param == 'byyear':
    #     plt.title(
    #         'Comparison of Regression Models Based on Best Parameter Values. Predicting experiment #2 on data from experiment #1')
    # elif year_param == 'byyearreveresed':
    #     plt.title(
    #         'Comparison of Regression Models Based on Best Parameter Values. Predicting experiment #1 on data from experiment #2')
    # else:
    #     plt.title('Comparison of Regression Models Based on Best Parameter Values')
    # plt.xlabel('Algorithm Name')
    # plt.ylabel('RMSE on test subset, cm²')
    # plt.legend(title='Regression Model', loc='lower right')
    # plt.grid(axis='y', zorder=0)
    #
    # # Add RMSE values on the bars
    # for p in ax.patches:
    #     height = p.get_height()
    #     ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
    #                 ha='center', va='bottom', fontsize=10, color='black', zorder=4)
    #
    # # Save the plot with an abstract name
    # plot_filename = f"{year_param}_RMSE_Best_per_Algorithm"
    # plot_filepath = os.path.join(plots_all_reconstructions_folder_path, plot_filename)
    # plt.savefig(plot_filepath)
    # plt.close()
    #
    # # Change axes to compare reconstruction methods
    # pivot_data = best_params.pivot(index='regression_model', columns='algorithm_name', values='RMSE_global_test')
    #
    # # Plot 2: Comparison of Reconstruction Methods
    # plt.figure(figsize=fig_size)
    # ax = pivot_data.plot(kind='bar', figsize=fig_size, zorder=3)
    # if year_param == 'byyear':
    #     plt.title('Comparison of Reconstruction Methods. Predicting experiment #2 on data from experiment #1')
    # elif year_param == 'byyearreveresed':
    #     plt.title('Comparison of Reconstruction Methods. Predicting experiment #1 on data from experiment #2')
    # else:
    #     plt.title('Comparison of Reconstruction Methods')
    # plt.xlabel('Regression Model')
    # plt.ylabel('RMSE on test subset, cm²')
    # plt.grid(axis='y', zorder=0)
    #
    # # Add R² values on the bars
    # for container in ax.containers:
    #     for bar in container:
    #         height = bar.get_height()
    #         ax.annotate(f'{height:.2f}', (bar.get_x() + bar.get_width() / 2., height),
    #                     ha='center', va='bottom', fontsize=10, color='black', zorder=4)
    #
    # plt.legend(title='Algorithm Name', loc='lower right')
    # plt.xticks(rotation=0)  # Set x-axis labels to be horizontal
    #
    # # Save the plot with an abstract name
    # plot_filename = f"{year_param}_RMSE_Comparison_Reconstruction_Methods"
    # plot_filepath = os.path.join(plots_all_reconstructions_folder_path, plot_filename)
    # plt.savefig(plot_filepath)
    # plt.close()

