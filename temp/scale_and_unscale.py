import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from config import paths

def split_data(df, feature_selection=True):
    # Split dataset into train and test using 'experiment_number' value. 1 for exp 1, 2 for exp 2
    train_df = df[df['experiment_number'] == 1].drop(columns=['experiment_number'])
    test_df = df[df['experiment_number'] == 2].drop(columns=['experiment_number'])

    # Prepare training and test data
    X_train = train_df.drop(columns=['measured_leaf_area'])
    y_train = train_df['measured_leaf_area']
    X_test = test_df.drop(columns=['measured_leaf_area'])
    y_test = test_df['measured_leaf_area']
    selected_features_mi = []

    if feature_selection:
        # Feature selection using Mutual Information
        mi_scores = mutual_info_regression(X_train, y_train)
        mi_scores = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)
        selected_features_mi = mi_scores[mi_scores > 0.1].index  # Adjust the threshold as needed
        X_train = X_train[selected_features_mi]
        X_test = X_test[selected_features_mi]

    # Reset indices to avoid row mismatching
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, y_train, X_test, y_test, selected_features_mi

folder_paths = paths.get_paths()
dataset_folder_path = folder_paths['ready_for_training']
test_folder_path = folder_paths['global_test_sets']

# Model parameters
algorithm_name = "alpha"
parameter_value = "3"
assessment_name = "trimesh"
dataset_type = "simple"

dataset_file_name = f"{algorithm_name}_{parameter_value}_{assessment_name}_{dataset_type}_noElim.csv"
dataset_file_path = os.path.join(dataset_folder_path, dataset_file_name)
df = pd.read_csv(dataset_file_path)

# Create and fit a scaler on the entire DataFrame except 'experiment_number'
scaler = StandardScaler()
df_scaled = df.copy()
scaling_columns = df.columns.difference(['experiment_number'])
df_scaled[scaling_columns] = scaler.fit_transform(df[scaling_columns])

print("Original DataFrame:\n", df.head(10).to_string())
print("Scaled DataFrame:\n", df_scaled.head(10).to_string())

# Split the data and perform feature selection
X_train, y_train, X_test, y_test, selected_features = split_data(df_scaled, feature_selection=True)
print("Shapes:", df_scaled.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

print(selected_features)

# Add missing columns with zeros and re-order columns
def add_missing_columns_and_reorder(df, original_columns):
    for col in original_columns:
        if col not in df.columns:
            df[col] = 0
    return df[original_columns]

X_train_full = add_missing_columns_and_reorder(train_df, df.columns.difference(['experiment_number']))
X_test_full = add_missing_columns_and_reorder(test_df, df.columns.difference(['experiment_number']))

# Unscale the selected features
train_unscaled = pd.DataFrame(scaler.inverse_transform(X_train_full), columns=X_train_full.columns)
test_unscaled = pd.DataFrame(scaler.inverse_transform(X_test_full), columns=X_test_full.columns)


print("Unscaled Train DataFrame:\n", train_unscaled.head(10).to_string())
print("Unscaled Test DataFrame:\n", test_unscaled.head(10).to_string())
print("Scaler parameters:")
print("Scale:", scaler.scale_)
print("Mean:", scaler.mean_)
print("Variance:", scaler.var_)
print("Number of features:", scaler.n_features_in_)
print("Feature names:", scaler.feature_names_in_)
