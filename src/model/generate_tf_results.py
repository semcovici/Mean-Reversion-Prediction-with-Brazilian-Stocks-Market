# https://drlee.io/advanced-stock-pattern-prediction-using-lstm-with-the-attention-mechanism-in-tensorflow-a-step-by-143a2e8b0e95

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import gc
from tensorflow.python.client import device_lib 
# Set random seed for reproducibility
tf.random.set_seed(42)
from keras.utils import to_categorical

import sys

sys.path.append("src/")
from model.evaluation import create_results_df
from model.tf_models import create_model_LSTM_with_Attention, create_model_MLP
from model.config import create_experiment_configs_tf
from data.preparation import load_dataset,prepare_data 



# Configuration
DATA_DIR = 'data/'
PATH_REPORTS = 'reports/'
PATH_MODELS = 'models/'
PATH_LOGS = "logs"

ASSETS = [
    "PETR3.SA", 
    "PRIO3.SA", 
    "VALE3.SA", 
    "GGBR3.SA", 
    "ABCB4.SA", 
    "ITUB3.SA", 
    "FLRY3.SA", 
    "RADL3.SA"
    ]

seq_len_list = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,14,21,28,35,42,49,56,63,70
    ]

dict_experiments = {}

exp_id = 0

moving_windows = [
    7,
    14,
    21
    ]

                            
dict_experiments = create_experiment_configs_tf(ASSETS, seq_len_list, moving_windows)

check_if_already_exists = True

def main():
   
    print(device_lib.list_local_devices())
    print("TensorFlow Version: ", tf.__version__)
    
    progress = 0
    for exp_name, config in dict_experiments.items():
        
        progress += 1
        
        print(f"""
#####################################
Running {progress + 1}/{len(dict_experiments)}
Config:
{config}
#####################################
""")
        
        feature_cols = config['feature_cols']
        label_col = config['label_col']
        seq_len = config['seq_len']
        asset = config['asset']
        scaling_method = config['scaling_method']
        algorithm = config['algorithm']
        prediction_type = config['prediction_type']
        

        path_results = PATH_REPORTS + f'test_results/{algorithm}_{asset.replace(".", "_")}_features={"_".join(feature_cols)}__label={label_col}__sql_len={seq_len}__scaling_method={scaling_method.__str__()}_test_results.csv'
        
        if os.path.isfile(path_results) and check_if_already_exists:
            print('# experiment already done')
            continue
        
        
        dataset = load_dataset(asset, DATA_DIR)
        X_train, X_test, y_train, y_test = prepare_data(dataset, seq_len, feature_cols, label_col, scaling_method)
        
        
        if prediction_type=='regression':
            
            num_classes = None
            
            callbacks = [
                EarlyStopping(monitor='val_r2_score', patience=10, mode='max'),
                # ModelCheckpoint(PATH_MODELS + f'best_model_LSTM_with_Attention_{asset.replace(".", "_")}.keras', save_best_only=True, monitor='val_r2_score'),
                ReduceLROnPlateau(monitor='val_r2_score', factor=0.1, patience=5, mode='max'),
                TensorBoard(log_dir=PATH_LOGS),
                CSVLogger(PATH_MODELS + f'training_log_{algorithm}_{asset.replace(".", "_")}_features={"_".join(feature_cols)}__label={label_col}__sql_len={seq_len}_test_results.csv')
            ]       
        elif prediction_type=='classification':
            num_classes = len(np.unique(y_train))
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, mode = 'min'),
                # ModelCheckpoint(PATH_MODELS + f'best_model_LSTM_with_Attention_{asset.replace(".", "_")}.keras', save_best_only=True, monitor='val_r2_score'),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,mode = 'min'),
                TensorBoard(log_dir=PATH_LOGS),
                CSVLogger(PATH_MODELS + f'training_log_{algorithm}_{asset.replace(".", "_")}_features={"_".join(feature_cols)}__label={label_col}__sql_len={seq_len}_test_results.csv')
            ]
            
            # Convertendo os rótulos para one-hot encoding
            y_train = to_categorical(y_train, num_classes=num_classes)
            y_test = to_categorical(y_test, num_classes=num_classes)

            
        else: raise ValueError(f'Não existe prediction_type = {prediction_type}')
            
        
        if algorithm == 'LSTM_with_Attention':
            model = create_model_LSTM_with_Attention(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                num_classes=num_classes
                )
        elif algorithm == 'MLP':
            model = create_model_MLP(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                num_classes=num_classes
                )
            
        else: raise ValueError(f'Não existe algorithm = {algorithm}') 
        
        print(model.summary())
    
    
        model.fit(X_train, y_train, epochs=1000, batch_size=64, validation_split=0.2, callbacks=callbacks)
        
        y_pred = model.predict(X_test)
        
        
        if prediction_type == 'regression':
        
            y_test = list(y_test.reshape(-1))
            y_pred = list(y_pred.reshape(y_pred.shape[0]))
        elif prediction_type == 'classification':
            
            # Convertendo os rótulos verdadeiros (y_test) de one-hot encoding para rótulos de classe
            y_test = np.argmax(y_test, axis=1)
            # Convertendo as previsões (y_pred) de probabilidades para rótulos de classe
            y_pred = np.argmax(y_pred, axis=1)
            # Transformando os arrays em listas para criar o DataFrame de resultados
            y_test = list(y_test.reshape(-1))
            y_pred = list(y_pred.reshape(-1))
            
        else: raise ValueError(f'Não existe prediction_type = {prediction_type}')
            

        results_df = create_results_df(y_test, y_pred)
        
        print("results in ", path_results)
        results_df.to_csv(path_results, index = False)
        
        gc.collect()
        tf.keras.backend.clear_session()

if __name__ == "__main__":
    main()
