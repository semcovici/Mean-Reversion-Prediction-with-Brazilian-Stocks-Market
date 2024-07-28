from keras.layers import Input, LSTM, Dense, Dropout, AdditiveAttention, Permute, Reshape, Multiply, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, AdditiveAttention, Permute, Reshape, Multiply, BatchNormalization, Input, Flatten
import keras

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

def create_model_MLP(input_shape):
    """Create and compile the MLP model."""
    inputs = Input(shape=(input_shape[0], input_shape[1]))
    x = Flatten()(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=METRICS)

    return model

def create_model_LSTM_with_Attention(input_shape):
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