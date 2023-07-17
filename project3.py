import os
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Tensorflow 2.x kann das wohl nicht, aber mit der compat funktion, kann man über den Profiler die flops ermitteln
# https://github.com/tensorflow/tensorflow/issues/32809
def get_flops(model):
    # Um den Tensorflow Graphen aus dem Keras Modell zu extrahieren, muss hier eine Umwandlung in eine
    # Tensorflow Funktion stattfinden. 
    func = tf.function(lambda x: model(x))
    concrete_func = func.get_concrete_function([tf.TensorSpec([1, *model.input_shape[1:]])])
    frozen_func = convert_variables_to_constants_v2(concrete_func)

    run_meta = tf.compat.v1.RunMetadata()

    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                          run_meta=run_meta,
                                          cmd='op',
                                          options=opts)

    return flops.total_float_ops

def scale_data(array):
    scaler = MaxAbsScaler()
    return scaler.fit_transform(np.array(array).reshape(-1, 1)).flatten()

def get_performances_normalized(flops, errors, lambda_):
    flops_norm = scale_data(flops)
    errors_norm = scale_data(errors)
    
    performances = [lambda_ * (1-flop) + (1 - lambda_) * (1-error) for error, flop in zip(errors_norm, flops_norm)]
    
    return performances

def generate_line_sequence_data(sequence_length, num_sequences, noise=0.1):
    X = np.empty((num_sequences, sequence_length))
    y = np.empty(num_sequences)
    
    for i in range(num_sequences):
        slope = np.random.uniform(-1, 1)
        intercept = np.random.uniform(-1, 1)
        x = np.linspace(-1, 1, sequence_length)
        y[i] = slope * (x[-1] + 1/sequence_length) + intercept  # Next point on the line
        X[i] = slope * x + intercept + np.random.normal(scale=noise, size=sequence_length)  # Line with noise

    return X[..., np.newaxis], y  # Add an extra dimension to X for the single feature

# Generate the data
X_train, y_train = generate_line_sequence_data(sequence_length=100, num_sequences=1000)
X_test, y_test = generate_line_sequence_data(sequence_length=100, num_sequences=200)


# Hier werden 3 CNNs erzeugt, mit untschiedlicher Anzahl an Layern und features 
def create_cnn_models(input_shape=(128, 128, 3)):
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
    
    return (model1,"conv-32-relu"),(model2,"conv-32-32-relu"),(model2_sigmoid,"conv-32-32-sigmoid"),(model3,"conv-32-64-relu")

def create_LSTM_models(): 
    # Model 1: Single LSTM layer with 128 units
    model1 = tf.keras.Sequential([
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(1)
    ])

    # Model 2: Two LSTM layers with 64 and 32 units
    model2 = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])

    model3 = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])



def train_and_evaluate_models(models, X_train, X_test, y_train, y_test, load=False):
    
    flops = []
    errors = []
    for model_label in models:
        model, lable = model_label
        print(f"Training model {lable}...")
        if load:
            model = load_model(f'modell-{lable}.h5')  
            
            with open(f'trainHistoryDict{lable}', 'rb') as file_pi:  
                history = pickle.load(file_pi)    
        else:
            opt = Adam(learning_rate=0.001)  
            model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='val_loss', patience=2)  
            history = model.fit(X_train, y_train, epochs=10, 
                                validation_data=(X_test, y_test),
                                callbacks=[early_stopping]).history
            
            model.save(f'modell-{lable}.h5')
            with open(f'trainHistoryDict{lable}', 'wb') as file_pi:
                pickle.dump(history, file_pi)
           
        
        print(f"Evaluating model {lable}...")
        flops.append(get_flops(model))
        errors.append(model.evaluate(X_test, y_test, verbose=1)[0] )
        

    return errors, flops

def get_train_test():
    folder = "./coil-100"  
    images, labels = load_images_from_folder(folder)
    images = np.array(images) / 255.0  
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    y_train = to_categorical(y_train, num_classes=100)
    y_test = to_categorical(y_test, num_classes=100)

    return X_train, X_test, y_train, y_test

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            label = filename.split("__")[0]  
            labels.append(label)
    return images, labels



def estimate_flops(model):
    # Es wird durch jede Schicht (layer) des Modells (model) iteriert.
# Wenn die Schicht eine Convolutional Schicht ist, müssen kernel size und kanäle berücksichtigt werden:
    # a. W_out, H_out und C_out sind die Breite, Höhe und Anzahl der Kanäle der Ausgabe der Schicht.
    # b. C_in ist die Anzahl der Kanäle der Eingabe der Schicht.
    # c. Die größe des Kernels wird anhand der Höhe und breite berechnet
    # d. Die Anzahl der FLOPs für diese Schicht wird mit 2 * kernel_size * C_in * C_out * W_out * H_out berechnet und zu total_flops hinzugefügt.
    # Die Ausgaber ist das Ergebnis einer Faltung, und jede Faltung beinhaltet kernel_size * C_in Multiplikationen und ebenso viele Additionen - 1
    # (da es nur eine Schätzung sein soll, ist die -1 denke ich vernachlässigbar, daher einfach mit 2 multiplizieren)
    # und dies wird für jede der C_out * W_out * H_out Ausgaben wiederholt.

# Beispiel:
# Es gibt ein 3x3-Eingangsbild (und Anzahl der Kanäle ist C_in = 1) 
# und einen 2x2-Kernel:

# Eingangsbild:
# 1 2 3
# 4 5 6
# 7 8 9

# Kernel:
# -1 0
#  0 1

# Eine Faltungsoperation in diesem Fall würde 4 Multiplikationen und 3 Additionen beinhalten 
# (insgesamt 7 Operationen), berechnet als:

# -1*1 + 0*2 + 0*4 + 1*5


# Über alle 2x2-Bereiche verschoben, führt das zu einem 2x2-Ausgabebild (W_out = H_out = 2):

# -1*2 + 0*3 + 0*5 + 1*6
# -1*4 + 0*5 + 0*7 + 1*8
# -1*5 + 0*6 + 0*8 + 1*9 

#  4 4
#  4 4

# Mit W_out = H_out = 2 ergeben das 2*2*7 = 28 Operationen.

# Wenn die Schicht eine Dense (voll verbundene) Schicht ist, wird die folgende Berechnung durchgeführt:
    # a. N_in und N_out sind die Anzahl der Neuronen in der Eingabe bzw. Ausgabe der Schicht.
    # b. Die Anzahl der FLOPs für diese Schicht wird als 2 * N_in * N_out berechnet und zu total_flops hinzugefügt.
    # Ähnlich wie in der Convolution Schicht mit weniger kernels und weniger Kanälen 

# Pooling und Ähnliches könnte man auch noch mit in die Schätzung aufnehmen, macht aber nur ca. 1-2% der flops aus
# Beispiel aus dem Profiler:
# Profile:
# node name | # float_ops
# Conv2D                   164.60m float_ops (100.00%, 91.01%)
# MatMul                   14.77m float_ops (8.99%, 8.17%)
# BiasAdd                  746.40k float_ops (0.82%, 0.41%)
# MaxPool                  738.43k float_ops (0.41%, 0.41%)
# Softmax                    500 float_ops (0.00%, 0.00%)
    total_flops = 0
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            print(layer.output_shape)
            _, W_out, H_out, C_out = layer.output_shape
            _, _, _, C_in = layer.input_shape
            kernel_size = layer.kernel_size[0] * layer.kernel_size[1] 
            total_flops += 2 * kernel_size * C_in * C_out * W_out * H_out
        elif isinstance(layer, Dense):
            N_in, N_out = layer.input_shape[-1], layer.output_shape[-1]
            total_flops += 2 * N_in * N_out
        elif isinstance(layer, keras.layers.LSTM):
            # Für eine LSTM layer, ist es ungefähr
            # 4 * (size_of_input + size_of_output) * (size_of_output) * sequence_length
            
            size_of_input = layer.input_shape[-1]
            size_of_output = layer.output_shape[-1]
            sequence_length = layer.input_shape[-2] 
            total_flops += 4 * (size_of_input + size_of_output) * size_of_output * sequence_length
    return total_flops

def calculate_flops_bounds(models):
    flops_values = [estimate_flops(model) for model in models]
    return min(flops_values), max(flops_values)

model1, model2, model2_sigmoid, model3 = create_cnn_models()


models = [model1, model2,  model2_sigmoid, model3]  
  
# Beispielwerte für Normalisierung und Lambda
# Für min_flops, max_flops muss noch eine gute Schätzung gefunden werden 
# min_flops, max_flops = calculate_flops_bounds(models)
# print(min_flops)
# print(max_flops)

min_error, max_error = 0.0, 1.0
# Über das Lambda kann gesteuert werden, ob der Error oder die FLOPs eine
# höhere Gewichtung für die Performance hat.
# bspw. lambda_ = 0.8 wenn die Performance für ein System mit niedriger Leistung evaluiert werden soll
lambda_ = 0.2

X_train, X_test, y_train, y_test = get_train_test()


model_comparison_pages = PdfPages('model_comparison_detailed.pdf')
errors, flops = train_and_evaluate_models(models, X_train, X_test, y_train, y_test, True)
# Plottet die Modellleistungen
for lambda_ in np.arange(0, 1, 0.1):
    print(f"evalutating with {lambda_} bias")
    performances = get_performances_normalized(flops,errors,lambda_)
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(performances) + 1), performances)
    plt.xlabel('Model Number')
    plt.ylabel('Performance')
    plt.title(f'Performance of Each Model with {lambda_} flops bias')
    plt.savefig(model_comparison_pages, format='pdf')

# plt.show()
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle(f'Errors and Flops of Each Model')

axs[0, 0].bar(range(1, len(errors) + 1), errors)
axs[0, 0].set_title('Errors')

axs[0, 1].bar(range(1, len(flops) + 1), flops)
axs[0, 1].set_title('FLOPs')

axs[1, 0].bar(range(1, len(scale_data(errors)) + 1), scale_data(errors))
axs[1, 0].set_title('Scaled Errors')

axs[1, 1].bar(range(1, len(scale_data(flops)) + 1), scale_data(flops))
axs[1, 1].set_title('Scaled FLOPs')

# Get the labels from the models tuple
labels = [model[1] for model in models]

# Set the x-axis tick labels
axs[0, 0].set_xticklabels(labels)
axs[0, 1].set_xticklabels(labels)
axs[1, 0].set_xticklabels(labels)
axs[1, 1].set_xticklabels(labels)

for ax in axs.flat:
    ax.set(xlabel='Model Number', ylabel='Value')

plt.savefig(model_comparison_pages, format='pdf')
model_comparison_pages.close()

performance_df = pd.DataFrame(performances, columns=['Performance'])
performance_df.to_csv('model_performances.csv', index=False)

#Fragen, die man in der Präsenation vorstellen sollten:
# Wie funktioniert der Profiler intern?
# Test mit verschiedenen Netzwerken: Feedforward, Recurrent, etc
# Abschätzung wie viele FLOPS könnte ein Netzwerk haben. Welchen Einfluss haben Aktivierungsfunktionen auf die FLOPs, Sigmoid, RELU.
# Innerhalb einer halben Stunde sollte es präsentiert werden können, jedes Gruppenmitglied soll einen Teil vorstellen können.