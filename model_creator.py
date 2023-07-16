from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LSTM
from keras.models import load_model


class ModelCreator:
    def __init__(self):
        pass

    # Hier werden 3 CNNs erzeugt, mit untschiedlicher Anzahl an Layern und features 
    def create_cnn_models(self,input_shape=(128, 128, 3)):
        model1 = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(100, activation='softmax')
        ])

        model2 = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(100, activation='softmax')
        ])

        model2_sigmoid = Sequential([
            Conv2D(32, (3, 3), activation='sigmoid', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, (3, 3), activation='sigmoid', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='sigmoid'),
            Dropout(0.5),
            Dense(100, activation='softmax')
        ])

        model3 = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(100, activation='softmax')
        ])
        
        return [(model1,"conv-32-relu"),(model2,"conv-32-32-relu"),(model2_sigmoid,"conv-32-32-sigmoid"),(model3,"conv-32-64-relu")]

    def create_lstm_models(self):
        # Model 1: Single LSTM layer with 128 units
        model1 = Sequential([
            LSTM(128),
            Dense(1)
        ])

        # Model 2: Two LSTM layers with 64 and 32 units
        model2 = Sequential([
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(1)
        ])

        model3 = Sequential([
            LSTM(32, return_sequences=True),
            LSTM(32),
            Dense(1)
        ])

        return [(model1,"lstm-128"),(model2,"lstm-64-32"),(model3,"lstm-64-32-32")]

    def load_model_from_system(self, lable):
        model = load_model(f'model-{lable}.h5')  
        return model