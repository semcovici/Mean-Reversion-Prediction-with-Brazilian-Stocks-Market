# https://drlee.io/advanced-stock-pattern-prediction-using-lstm-with-the-attention-mechanism-in-tensorflow-a-step-by-143a2e8b0e95

import os
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout, AdditiveAttention, Permute, Reshape, Multiply, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, AdditiveAttention, Permute, Reshape, Multiply, BatchNormalization, Input, Flatten
from tensorflow.python.client import device_lib 
# Set random seed for reproducibility
tf.random.set_seed(42)

# Configuration
DATA_DIR = 'data/'
PATH_REPORTS = 'reports/'
PATH_MODELS = 'models/'
PATH_LOGS = "logs"

ASSETS = ["PETR3.SA", "PRIO3.SA", "VALE3.SA", "GGBR3.SA", "ABCB4.SA", "ITUB3.SA", "FLRY3.SA", "RADL3.SA"]
RELEVANT_COLS = ['Date', 'Close', 'Volume']
SEQ_LEN = 60
FEATURES_COLS = ['diff_close_mean_z_score']

METRICS = [
    #keras.metrics.F1Score(average='macro',name='f1_score'),
    keras.metrics.BinaryCrossentropy(name='cross entropy'),  # same as model's loss
      
      
      # regression metrics
      keras.metrics.R2Score(name = 'r2_score'),
      keras.metrics.MeanSquaredError(name='mse'),
      keras.metrics.MeanAbsolutePercentageError(name='mape')

      # classification metrics      
      # keras.metrics.TruePositives(name='tp'),
      # keras.metrics.FalsePositives(name='fp'),
      # keras.metrics.TrueNegatives(name='tn'),
      # keras.metrics.FalseNegatives(name='fn'), 
      # keras.metrics.BinaryAccuracy(name='accuracy'),
      # keras.metrics.Precision(name='precision'),
      # keras.metrics.Recall(name='recall'),
      # keras.metrics.AUC(name='auc'),
      # keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

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

# def create_model(input_shape):
#     """Create and compile the LSTM model."""
#     model = Sequential()
#     model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
#     model.add(LSTM(units=50, return_sequences=True))
    
#     attention = AdditiveAttention(name='attention_weight')
#     model.add(Permute((2, 1))) 
#     model.add(Reshape((-1, input_shape[0])))
#     attention_result = attention([model.output, model.output])
#     multiply_layer = Multiply()([model.output, attention_result])
#     model.add(Permute((2, 1))) 
#     model.add(Reshape((-1, 50)))

#     model.add(tf.keras.layers.Flatten())
#     model.add(Dense(1))
#     model.add(Dropout(0.2))
#     model.add(BatchNormalization())

#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=METRICS)

#     return model

def create_model(input_shape):
    """Create and compile the LSTM model."""
    inputs = Input(shape=input_shape)
    x = LSTM(units=50, return_sequences=True)(inputs)
    x = LSTM(units=50, return_sequences=True)(x)
    
    attention = AdditiveAttention(name='attention_weight')([x, x])
    x = Multiply()([x, attention])
    x = Flatten()(x)
    x = Dense(1)(x)
    x = Dropout(0.2)(x)
    outputs = BatchNormalization()(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=METRICS)

    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    test_loss = model.evaluate(X_test, y_test)
    print("Test Loss: ", test_loss)

    y_pred = model.predict(X_test)
    print(classification_report([int(i) for i in y_test], [int(i) for i in y_pred]))
    
def create_results_df(y_test, y_pred):
    
    results_df = pd.DataFrame({
        'y_test': y_test,
        'y_pred': y_pred
    })
    
    return results_df
    
    


    

def main():
   
    print(device_lib.list_local_devices())
    print("TensorFlow Version: ", tf.__version__)

    for asset in ASSETS:
        dataset = load_dataset(asset, DATA_DIR)
        X_train, X_test, y_train, y_test = prepare_data(dataset, SEQ_LEN, FEATURES_COLS)
        
        model = create_model((X_train.shape[1], X_train.shape[2]))
        print(model.summary())

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10),
            ModelCheckpoint(PATH_MODELS + f'best_model_LSTM_with_Attention_{asset.replace('.', '_')}.keras', save_best_only=True, monitor='val_loss'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
            TensorBoard(log_dir=PATH_LOGS),
            CSVLogger(PATH_MODELS + f'training_log_LSTM_with_Attention_{asset.replace('.', '_')}.csv')
        ]

        model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2, callbacks=callbacks)
        
        y_pred = model.predict(X_test)
        
        y_test = list(y_test)
        y_pred = list(y_pred.reshape(y_pred.shape[0]))
        results_df = create_results_df(y_test, y_pred)
        
        results_df.to_csv(PATH_REPORTS + f'test_results/LSTM_with_Attention_{asset.replace('.', '_')}_test_results.csv')

if __name__ == "__main__":
    main()
