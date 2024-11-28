from keras.layers import Input, LSTM, Dense, Dropout, AdditiveAttention, Permute, Reshape, Multiply, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, AdditiveAttention, Permute, Reshape, Multiply, BatchNormalization, Input, Flatten, Bidirectional
import keras
from kan import KAN
import torch

seed = 42

METRICS_CLF = [
    #keras.metrics.F1Score(average='macro',name='f1_score'),
    keras.metrics.BinaryCrossentropy(name='cross entropy'),  # same as model's loss
      
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]


METRICS_REG = [
    #keras.metrics.F1Score(average='macro',name='f1_score'),
    keras.metrics.BinaryCrossentropy(name='cross entropy'),  # same as model's loss
      # regression metrics
      keras.metrics.R2Score(name = 'r2_score'),
      keras.metrics.MeanSquaredError(name='mse'),
      keras.metrics.MeanAbsolutePercentageError(name='mape')

]


def create_model_MLP(input_shape, num_classes=None):
    """Create and compile the MLP model."""
    inputs = Input(shape=(input_shape[0], input_shape[1]))
    x = Flatten()(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Se for classificação, n neurônios de saída com softmax
    if num_classes:  # num_classes indica que é uma tarefa de classificação
        metrics = METRICS_CLF
        outputs = Dense(num_classes, activation='softmax')(x)
        loss = 'categorical_crossentropy'  # Ou sparse_categorical_crossentropy, dependendo dos rótulos
    else:  # Caso contrário, é uma tarefa de regressão
        metrics = METRICS_REG
        outputs = Dense(1)(x)
        loss = 'mean_squared_error'
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    
    return model


# https://drlee.io/advanced-stock-pattern-prediction-using-lstm-with-the-attention-mechanism-in-tensorflow-a-step-by-143a2e8b0e95
def create_model_LSTM_with_Attention(input_shape, num_classes=None):
    """Create and compile the LSTM model."""    
    inputs = Input(shape=input_shape)
    
    x = Bidirectional(LSTM(units=50, return_sequences=True))(inputs)
    x = Bidirectional(LSTM(units=50, return_sequences=True))(x)
        
    attention = AdditiveAttention(name='attention_weight')([x, x])
    x = Multiply()([x, attention])
    x = Flatten()(x)
    x = Dropout(0.2)(x)

    # Se for classificação, n neurônios de saída com softmax
    if num_classes:  # num_classes indica que é uma tarefa de classificação
        metrics = METRICS_CLF
        outputs = Dense(num_classes, activation='softmax')(x)
        loss = 'categorical_crossentropy'  # Ou sparse_categorical_crossentropy, dependendo dos rótulos
    else:  # Caso contrário, é uma tarefa de regressão
        metrics = METRICS_REG
        outputs = Dense(1)(x)
        loss = 'mean_squared_error'

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss=loss, metrics=metrics)

    return model


def create_model_KAN(input_shape, num_classes=None):
    """Create the KAN model."""
    
    # Captura o device do torch para executar o modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Se for classificação, n neurônios de saída com softmax
    if num_classes: # num_classes indica que é uma tarefa de classificação
        model = KAN(width=[input_shape,40,40,40,num_classes], grid=5, k=3, seed=seed)
        model.to(device)
        
    else: # Caso contrário, é uma tarefa de regressão
        model = KAN(width=[input_shape,40,40,40,1], grid=5, k=3, seed=seed)
        model.to(device)
        
    return model