import numpy as np
from sklearn.preprocessing import MaxAbsScaler
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow import keras
from keras.layers import Conv2D, Dense, LSTM

class ModelEvaluator:
    def __init__(self):
        pass


    def get_flops(self,model):
        # Um den Tensorflow Graphen aus dem Keras Modell zu extrahieren, muss hier eine Umwandlung in eine
        # Tensorflow Funktion stattfinden. 
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
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

    def scale_data(self,array):
        scaler = MaxAbsScaler()
        return scaler.fit_transform(np.array(array).reshape(-1, 1)).flatten()

    def evaluate_model(self, model, X_test, y_test):
        flops = self.get_flops(model)
        error = model.evaluate(X_test, y_test, verbose=1)[0]
        estimated_flops = self.estimate_flops(model)
        return error, flops, estimated_flops

    def get_performances_normalized(self, flops, errors, lambda_):
        flops_norm = self.scale_data(flops)
        errors_norm = self.scale_data(errors)
        
        performances = [lambda_ * (1-flop) + (1 - lambda_) * (1-error) for error, flop in zip(errors_norm, flops_norm)]
        
        return performances
    
    def estimate_flops(self,model):
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
        number_of_lstm_layers = len([layer for layer in model.layers if isinstance(layer, LSTM)])
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

            elif isinstance(layer, LSTM):
                size_of_input = layer.input_shape[-1]  
                size_of_output = layer.output_shape[-1]  
                sequence_length = layer.input_shape[1]  # timesteps
                
                total_flops += number_of_lstm_layers * sequence_length * 8 * 2 * size_of_output * (size_of_input + size_of_output)


        return total_flops
