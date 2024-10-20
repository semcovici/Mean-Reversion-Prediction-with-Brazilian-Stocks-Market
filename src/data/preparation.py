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

def prepare_data(
    dataset, 
    seq_len, 
    features_cols,
    label_col,
    scaling_method = None
    ):
    """Prepare training and testing data."""
    train_index = dataset[dataset.split == 'train'].index
    test_index = dataset[dataset.split == 'test'].index
    
    if scaling_method is not None:
        X = dataset[features_cols]
        X_train = X[X.index.isin(train_index)]
        X_test = X[X.index.isin(test_index)]
        
        scaler = scaling_method.fit(X_train)
        X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=features_cols)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features_cols)
        
        X = pd.concat([X_train_scaled, X_test_scaled]).values
        
        del X_train, X_test, X_train_scaled, X_test_scaled
    
    else:
        X = dataset[features_cols].values
        
    y = dataset[[label_col]].values

    X_train, X_test, y_train, y_test = [], [], [], []

    for i in range(seq_len, len(dataset)):
        if i in train_index: 
            X_train.append(X[i-seq_len:i])
            y_train.append(y[i])
        elif i in test_index:
            X_test.append(X[i-seq_len:i])
            y_test.append(y[i])
        else:
            raise ValueError('Value not found in index lists')

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    return X_train, X_test, y_train, y_test