import os
import pandas as pd
import numpy as np


def load_dataset(asset, data_dir):
    """Load train and test datasets for a given asset."""
    train_file = os.path.join(data_dir, f"processed/train_price_history_{asset.replace('.', '_')}_meta_dataset_ffill.csv")
    test_file = os.path.join(data_dir, f"processed/test_price_history_{asset.replace('.', '_')}_meta_dataset_ffill.csv")
    
    train_dataset = pd.read_csv(train_file, index_col=0).reset_index(drop=True)
    train_dataset['split'] = 'train'
    test_dataset = pd.read_csv(test_file, index_col=0).reset_index(drop=True)
    test_dataset['split'] = 'test'
    
    return pd.concat([train_dataset, test_dataset], ignore_index=True)

def prepare_data(dataset, seq_len, features_cols):
    """Prepare training and testing data."""
    train_index = dataset[dataset.split == 'train'].index
    test_index = dataset[dataset.split == 'test'].index

    dataset_values = dataset[features_cols].values

    X_train, X_test, y_train, y_test = [], [], [], []

    for i in range(seq_len, len(dataset_values)):
        if i in train_index: 
            X_train.append(dataset_values[i-seq_len:i])
            y_train.append(dataset_values[i, 0])
        elif i in test_index:
            X_test.append(dataset_values[i-seq_len:i])
            y_test.append(dataset_values[i, 0])
        else:
            raise ValueError('Value not found in index lists')

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    return X_train, X_test, y_train, y_test