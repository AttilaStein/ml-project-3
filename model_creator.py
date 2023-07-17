from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LSTM
from keras.models import load_model
from keras.utils import plot_model

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
        
        plot_model(model1, to_file='cnn_model1.png', show_shapes=True, show_layer_names=True)
        plot_model(model2, to_file='cnn_model2.png', show_shapes=True, show_layer_names=True)
        plot_model(model2_sigmoid, to_file='cnn_model2_sigmoid.png', show_shapes=True, show_layer_names=True)
        plot_model(model3, to_file='cnn_model3.png', show_shapes=True, show_layer_names=True)

        return [(model1,"conv-32-relu"),(model2,"conv-32-32-relu"),(model2_sigmoid,"conv-32-32-sigmoid"),(model3,"conv-32-64-relu")]

    def create_lstm_models(self, input_shape=(100, 1)):
        model1 = Sequential([
            LSTM(128, input_shape=input_shape),
            Dense(1)
        ])
        model1.build()

        model2 = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            LSTM(32),
            Dense(1)
        ])
        model2.build()

        model3 = Sequential([
            LSTM(32, return_sequences=True, input_shape=input_shape),
            LSTM(32),
            Dense(1)
        ])
        model3.build()
        plot_model(model1, to_file='lstm_model1.png', show_shapes=True, show_layer_names=True)
        plot_model(model2, to_file='lstm_model2.png', show_shapes=True, show_layer_names=True)
        plot_model(model3, to_file='lstm_model3.png', show_shapes=True, show_layer_names=True)

        return [(model1,"lstm-128"),(model2,"lstm-64-32"),(model3,"lstm-64-32-32")]

    def load_model_from_system(self, lable):
        model = load_model(f'model-{lable}.h5')  
        return model