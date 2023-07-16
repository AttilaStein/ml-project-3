
import os
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import to_categorical
import cv2
import pickle

class ModelTrainer:
    def __init__(self):
        pass

    def train_lstm(self, models_with_labels):
        print (models_with_labels)
        trained_models = []
        X_train, X_test, y_train, y_test = self.get_train_test_lstm()
        np.save('X_lstm_test.npy', X_test)
        np.save('y_lstm_test.npy', y_test)
        for model, label in models_with_labels:
            print("Training LSTM model...")
            
            opt = Adam(learning_rate=0.001)  
            model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_absolute_error'])
            early_stopping = EarlyStopping(monitor='val_loss', patience=2)  
            history = model.fit(X_train, y_train, epochs=10, 
                                validation_data=(X_test, y_test),
                                callbacks=[early_stopping]).history
            model.save(f'model-{label}.h5')
            with open(f'trainHistoryDict-lstm', 'wb') as file_pi:
                pickle.dump(history, file_pi)
            
            print("LSTM model trained.")
            trained_models.append(model)
            

        return trained_models, X_test, y_test

    def train_cnn(self, models_with_labels):
        X_train, X_test, y_train, y_test = self.get_train_test_cnn()
        np.save('X_cnn_test.npy', X_test)
        np.save('y_cnn_test.npy', y_test)
        trained_models = []
        for model, label in models_with_labels:

            print(f"Training model {label}...")
            opt = Adam(learning_rate=0.001)  
            model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='val_loss', patience=2)  
            history = model.fit(X_train, y_train, epochs=10, 
                                validation_data=(X_test, y_test),
                                callbacks=[early_stopping]).history
            
            model.save(f'modell-{label}.h5')
            with open(f'trainHistoryDict{label}', 'wb') as file_pi:
                pickle.dump(history, file_pi)
            
            print(f"Evaluating model {label}...")
            trained_models.append(model)

        return trained_models, X_test, y_test

    def load_images_from_folder(self, folder):
        images = []
        labels = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
                label = filename.split("__")[0]  
                labels.append(label)
        return images, labels


    def get_train_test_cnn(self):
        folder = "./coil-100"  
        images, labels = self.load_images_from_folder(folder)
        images = np.array(images) / 255.0  
        labels = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)

        y_train = to_categorical(y_train, num_classes=100)
        y_test = to_categorical(y_test, num_classes=100)

        return X_train, X_test, y_train, y_test

    def generate_line_sequence_data(self, sequence_length, num_sequences, noise=0.1):
        X = np.empty((num_sequences, sequence_length))
        y = np.empty(num_sequences)
        
        for i in range(num_sequences):
            slope = np.random.uniform(-1, 1)
            intercept = np.random.uniform(-1, 1)
            x = np.linspace(-1, 1, sequence_length)
            y[i] = slope * (x[-1] + 1/sequence_length) + intercept  # Next point on the line
            X[i] = slope * x + intercept + np.random.normal(scale=noise, size=sequence_length)  # Line with noise

        return X[..., np.newaxis], y  # Add an extra dimension to X for the single feature


    def get_train_test_lstm(self):
        X, y = self.generate_line_sequence_data(sequence_length=100, num_sequences=1000)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test

    def load_trained_model(self, label, model_type):
        model_path = f'model-{label}.h5'
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            print(f"Model {label} loaded from {model_path}")
            X_test = np.load(f'X_{model_type}_test.npy')
            y_test = np.load(f'y_{model_type}_test.npy')
            return model, X_test, y_test
        else:
            print(f"No saved model found for label {label}")
            return None


