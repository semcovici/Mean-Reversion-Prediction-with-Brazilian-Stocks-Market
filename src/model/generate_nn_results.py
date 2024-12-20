
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
import os
import tensorflow as tf
import numpy as np
import gc
from tensorflow.python.client import device_lib 
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import sys

sys.path.append("src/")
from model.evaluation import create_results_df
from model.nn_models import create_model_LSTM_with_Attention, create_model_MLP, create_model_KAN
from model.config import create_experiment_configs_tf
from data.preparation import load_dataset,prepare_data 

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Is cuda available: {torch.cuda.is_available()}")
if str(device) != 'cuda':
	raise ValueError('device must be cuda')

# Function to set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)     
    tf.random.set_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True # pesquisar melhor
    torch.backends.cudnn.benchmark = False

# Setting seed
seed=42
set_seed(seed)

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

moving_windows = [
    7,
    14,
    21
    ]
algorithms=[
    #'LSTM_with_Attention', 
    #'MLP'
     'KAN'
    ]
                            
dict_experiments = create_experiment_configs_tf(ASSETS, seq_len_list, moving_windows, algorithms)

check_if_already_exists = True

def main():
   
    print(device_lib.list_local_devices())
    print("TensorFlow Version: ", tf.__version__)
    
    progress = 0
    for exp_name, config in dict_experiments.items():
        
        progress += 1
        
        print(f"""
#####################################
Running {progress}/{len(dict_experiments)}
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
        X_train, X_valid, X_test, y_train, y_valid, y_test = prepare_data(dataset, seq_len, feature_cols, label_col, scaling_method,valid=True,valid_pct=0.1)
    
        if prediction_type=='regression':
            
            num_classes = None
            
            if algorithm in ['MLP', 'LSTM_with_Attention']:
                callbacks = [
                    EarlyStopping(monitor='val_r2_score', patience=10, mode='max'),
                    # ModelCheckpoint(PATH_MODELS + f'best_model_LSTM_with_Attention_{asset.replace(".", "_")}.keras', save_best_only=True, monitor='val_r2_score'),
                    ReduceLROnPlateau(monitor='val_r2_score', factor=0.1, patience=5, mode='max'),
                    TensorBoard(log_dir=PATH_LOGS),
                    CSVLogger(PATH_MODELS + f'training_log_{algorithm}_{asset.replace(".", "_")}_features={"_".join(feature_cols)}__label={label_col}__sql_len={seq_len}_test_results.csv')
                ]   
            elif algorithm == 'KAN':
                # nao sei se é necessario fazer algo aqui
                # preencher
                # tenho que verificar se a entrada sao arrays numpy
                pass
            
            else: raise ValueError(f'Algoritmo nao esperado - {algorithm}')
            
        elif prediction_type=='classification':
            
            # concatena treino e validacao para saber o conjunto de labels possiveis
            y_train_val = np.concatenate([y_valid, y_train]).reshape(-1)
            
            num_classes = len(np.unique(y_train_val))

            
            if algorithm in ['MLP', 'LSTM_with_Attention']:
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, mode = 'min'),
                    # ModelCheckpoint(PATH_MODELS + f'best_model_LSTM_with_Attention_{asset.replace(".", "_")}.keras', save_best_only=True, monitor='val_r2_score'),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,mode = 'min'),
                    TensorBoard(log_dir=PATH_LOGS),
                    CSVLogger(PATH_MODELS + f'training_log_{algorithm}_{asset.replace(".", "_")}_features={"_".join(feature_cols)}__label={label_col}__sql_len={seq_len}_test_results.csv')
                ]

                # Convertendo os rótulos para one-hot encoding #
                # cria label encoder
                le = LabelEncoder()
                le.fit(y_train_val)
                y_train_val_enc = le.transform(y_train_val)
                #cria one
                one = OneHotEncoder()
                one.fit(y_train_val_enc.reshape(-1,1))
                y_train_enc = le.transform(y_train.reshape(-1)).reshape(-1,1)
                y_train = one.transform(y_train_enc).toarray()
                y_valid_enc = le.transform(y_valid.reshape(-1)).reshape(-1,1)
                y_valid = one.transform(y_valid_enc).toarray()
                
            elif algorithm == 'KAN':
                # preencher
                # fazer processamento dos dados de rotulo que nao é pra estar em to_categorical,
                # mas sim em squeeze
                # e tbm tenho que verificar se entradas sao arrays numpy
                pass
                
            else: raise ValueError(f'Algoritmo nao esperado - {algorithm}')

            
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
            
        elif algorithm == 'KAN':

            y_train = y_train.squeeze()
            y_val = y_valid.squeeze()
            y_test = y_test.squeeze()

            # Realizando o flatten das séries de entrada
            X_train_flatten = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
            X_val_flatten = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1]*X_valid.shape[2]))
            X_test_flatten = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

            print(np.unique(y_train, return_counts=True))

            if prediction_type == 'classification':
                y_train_val = np.concatenate([y_train, y_val])
                le = LabelEncoder()
                le = le.fit(y_train_val)
                y_train = le.transform(y_train)  
                y_val =  le.transform(y_val)  

            
            y_dtype = torch.float if prediction_type == 'regression' else torch.long

            train_input = torch.from_numpy(X_train_flatten).type(torch.float32).to(device)
            train_label = torch.from_numpy(y_train).type(y_dtype).to(device)

            #print(train_input.shape)
            
            val_input = torch.from_numpy(X_val_flatten).type(torch.float32).to(device)
            val_label = torch.from_numpy(y_val).type(y_dtype).to(device)

            test_input = torch.from_numpy(X_test_flatten).type(torch.float32).to(device)
            test_label = torch.from_numpy(y_test).type(y_dtype).to(device) # tirar talvez dps
            
    
            model = create_model_KAN(
            input_shape=X_train_flatten.shape[1], # quantidade de elementos no array unidimensional
            num_classes=num_classes
            )
            
        else: raise ValueError(f'Não existe algorithm = {algorithm}') 
        
        #print(model.summary())
    
    
        if algorithm in ['MLP', 'LSTM_with_Attention']:
            
            print(y_train.shape)
            print(y_valid.shape)
            
            
            # model.fit(X_train, y_train, epochs=1000, batch_size=64, validation_split=0.2, callbacks=callbacks)
            model.fit(
    X_train, 
    y_train, 
    epochs=1000, 
    batch_size=64, 
    validation_data=(X_valid, y_valid), 
    callbacks=callbacks
)


            y_pred = model.predict(X_test)
        
        elif algorithm == 'KAN':

            # define the loss function 
            if prediction_type=='regression':

                loss_fn = torch.nn.MSELoss()

            elif prediction_type=='classification':

                loss_fn = torch.nn.CrossEntropyLoss()

            else: raise ValueError('PIPIPIPIP')


            # como lidar com train_input sendo que ele é um processamento de X_train, e ademais com outros
            model.fit({'train_input': train_input, 'train_label': train_label,
                        'test_input': val_input, 'test_label': val_label},
                        opt="Adam",
                        lr=0.001,
                        steps=150,
						lamb=0.001,
						lamb_l1=0.005,
						lamb_entropy=0.002,
                        loss_fn=loss_fn)

            y_pred = model.forward(test_input).detach()

            # define the loss function 
            #if prediction_type=='classification':

                #y_pred = torch.argmax(y_pred,dim=1)

            print(y_pred)
            print(y_pred.shape)         
                
            y_pred = y_pred.cpu().numpy()

            print(y_pred)
            print(y_pred.shape)

            # gerar as predicoes aqui, mas acho que o fit eu tenho que dar antes mesmo talvez (na parte de regressao ou classificacao)
            # qual sera que é o tipo da variavel de y_pred para MLP e LSTM? (verificar para ver se é igual)
        
        if prediction_type == 'regression':

                y_test = list(y_test.reshape(-1))
                y_pred = list(y_pred.reshape(y_pred.shape[0]))

        elif prediction_type == 'classification':

            # Convertendo os rótulos verdadeiros (y_test) de one-hot encoding para rótulos de classe
            #y_test = np.argmax(y_test, axis=1)
            # Convertendo as previsões (y_pred) de probabilidades para rótulos de classe
            y_pred = np.argmax(y_pred, axis=1)

            y_pred = le.inverse_transform(y_pred)

            # Transformando os arrays em listas para criar o DataFrame de resultados
            y_test = list(y_test.reshape(-1))
            y_pred = list(y_pred.reshape(-1))
        
        #if algorithm in ['MLP', 'LSTM']:
        
        #    if prediction_type == 'regression':

        #        y_test = list(y_test.reshape(-1))
        #        y_pred = list(y_pred.reshape(y_pred.shape[0]))
        #    elif prediction_type == 'classification':

                # Convertendo os rótulos verdadeiros (y_test) de one-hot encoding para rótulos de classe
        #        y_test = np.argmax(y_test, axis=1)
                # Convertendo as previsões (y_pred) de probabilidades para rótulos de classe
        #        y_pred = np.argmax(y_pred, axis=1)
                # Transformando os arrays em listas para criar o DataFrame de resultados
        #        y_test = list(y_test.reshape(-1))
        #        y_pred = list(y_pred.reshape(-1))
                
        #elif algorithm == 'KAN':
            
        #    if prediction_type == 'regression':

        #        y_test = list(y_test.reshape(-1))
                
        #    elif prediction_type == 'classification':

        #        y_test = list(y_test.reshape(-1))
            
            
        #else: raise ValueError(f'Não existe prediction_type = {prediction_type}')
            

        results_df = create_results_df(y_test, y_pred)
        
        print("results in ", path_results)
        results_df.to_csv(path_results, index = False)
        
        # Limpeza da sessão
        gc.collect()
        tf.keras.backend.clear_session()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
