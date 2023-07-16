import numpy as np
from sklearn.preprocessing import MaxAbsScaler
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

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
        return error, flops

    def get_performances_normalized(self, flops, errors, lambda_):
        flops_norm = self.scale_data(flops)
        errors_norm = self.scale_data(errors)
        
        performances = [lambda_ * (1-flop) + (1 - lambda_) * (1-error) for error, flop in zip(errors_norm, flops_norm)]
        
        return performances
