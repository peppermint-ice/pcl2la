import pandas as pd
import os

file_path = r"C:\Users\dusen\Documents\PhD\plys\data\ready_for_training\alpha_5_o3d_simple_noElim.csv"
df = pd.read_csv(file_path)

# Split dataset into train and test using 'experiment_number' value. 2023 for exp 1, 2024 for exp 2
train_df = df[df['experiment_number'] == 1].drop(columns=['experiment_number'])
test_df = df[df['experiment_number'] == 2].drop(columns=['experiment_number'])

train_df.to_csv('train_df.csv', index=False)
test_df.to_csv('test_df.csv', index=False)
