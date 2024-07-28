import os
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Input, Flatten
from tensorflow.python.client import device_lib 

import sys
sys.path.append("src/")
from model.evaluation import create_results_df
from model.tf_models import create_model_MLP
from data.preparation import load_dataset,prepare_data 

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
    keras.metrics.BinaryCrossentropy(name='cross entropy'),
    keras.metrics.R2Score(name='r2_score'),
    keras.metrics.MeanSquaredError(name='mse'),
    keras.metrics.MeanAbsolutePercentageError(name='mape')
]

def main():
    print(device_lib.list_local_devices())
    print("TensorFlow Version: ", tf.__version__)

    for asset in ASSETS:
        dataset = load_dataset(asset, DATA_DIR)
        X_train, X_test, y_train, y_test = prepare_data(dataset, SEQ_LEN, FEATURES_COLS)
        
        model = create_model_MLP((X_train.shape[1], X_train.shape[2]))
        print(model.summary())

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10),
            ModelCheckpoint(PATH_MODELS + f'best_model_MLP_{asset.replace('.', '_')}.keras', save_best_only=True, monitor='val_loss'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
            TensorBoard(log_dir=PATH_LOGS),
            CSVLogger(PATH_MODELS + f'training_log_MLP_{asset.replace('.', '_')}.csv')
        ]

        model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2, callbacks=callbacks)
        
        y_pred = model.predict(X_test)
        
        y_test = list(y_test)
        y_pred = list(y_pred.reshape(y_pred.shape[0]))
        results_df = create_results_df(y_test, y_pred)
        
        results_df.to_csv(PATH_REPORTS + f'test_results/MLP_{asset.replace('.', '_')}_test_results.csv')

if __name__ == "__main__":
    main()
