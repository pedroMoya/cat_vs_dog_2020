# Model architecture analyzer
import os
import logging
import logging.handlers as handlers
import json
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
tf.keras.backend.set_floatx('float32')
from tensorflow.keras import layers
from tensorflow.keras.experimental import PeepholeLSTMCell
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import losses, models
from tensorflow.keras import metrics
from tensorflow.keras import callbacks as cb
from tensorflow.keras import backend as kb
from sklearn.metrics import mean_squared_error
from tensorflow.keras.utils import plot_model, model_to_dot

# open local settings
with open('./settings.json') as local_json_file:
    local_submodule_settings = json.loads(local_json_file.read())
    local_json_file.close()

# log setup
current_script_name = os.path.basename(__file__).split('.')[0]
log_path_filename = ''.join([local_submodule_settings['log_path'], current_script_name, '.log'])
logging.basicConfig(filename=log_path_filename, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logHandler = handlers.RotatingFileHandler(log_path_filename, maxBytes=10485760, backupCount=5)
logger.addHandler(logHandler)

# function definitions

class customized_loss(losses.Loss):
    @tf.function
    def call(self, local_true, local_pred):
        softmax_diff = tf.math.abs(tf.math.add(tf.nn.log_softmax(local_true), -tf.nn.log_softmax(local_pred)))
        return softmax_diff


class customized_loss_auc_roc(losses.Loss):
    @tf.function
    def call(self, local_true, local_pred):
        local_true = tf.convert_to_tensor(local_true, dtype=tf.float32)
        local_pred = tf.convert_to_tensor(local_pred, dtype=tf.float32)
        local_auc_roc = tf.math.add(tf.math.add(1., metrics.AUC(local_true, local_pred)))
        return local_auc_roc


class customized_loss2(losses.Loss):
    @tf.function
    def call(self, local_true, local_pred):
        local_true = tf.convert_to_tensor(local_true, dtype=tf.float32)
        local_pred = tf.convert_to_tensor(local_pred, dtype=tf.float32)
        factor_difference = tf.reduce_mean(tf.abs(tf.add(local_pred, -local_true)))
        factor_true = tf.reduce_mean(tf.add(tf.convert_to_tensor(1., dtype=tf.float32), local_true))
        return tf.math.multiply_no_nan(factor_difference, factor_true)


# classes definitions


class model_structure:

    def analize(self, local_model_name, local_a_settings, local_a_hyperparameters):
        try:
            # loading model (h5 format)
            print('trying to open model file (assuming h5 format)')
            custom_obj = {'customized_loss': customized_loss, 'customized_loss_auc_roc': customized_loss_auc_roc}
            local_model = models.load_model(''.join([local_a_settings['models_path'], local_model_name]),
                                            custom_objects=custom_obj)
            # saving architecture in JSON format
            local_model_json = local_model.to_json()
            with open(''.join([local_a_settings['models_path'], local_model_name,
                               '_analyzed_.json']), 'w') as json_file:
                json_file.write(local_model_json)
                json_file.close()
            # changing for subclassing to functional model
            model_json = json.loads(local_model_json)
            local_batch_size = None
            local_y_input = local_a_hyperparameters['input_shape_y']
            local_x_input = local_a_hyperparameters['input_shape_x']
            local_nof_channels = local_a_hyperparameters['nof_channels']
            plot_path = ''.join([local_a_settings['models_path'], local_model_name, '_model.png'])
            if local_a_settings['use_efficientNetB2'] == 'False':
                input_layer = layers.Input(batch_shape=(local_batch_size, local_y_input, local_x_input,
                                                        local_nof_channels))
                prev_layer = input_layer
                for layer in local_model.layers:
                    prev_layer = layer(prev_layer)
                functional_model = models.Model([input_layer], [prev_layer])
                # plotting (exporting to png) the model
                # model_to_dot(functional_model, show_shapes=True, show_layer_names=True, rankdir='TB',
                #     expand_nested=True, dpi=96, subgraph=True)
                plot_model(functional_model, to_file=plot_path, show_shapes=True, show_layer_names=True,
                           rankdir='TB', expand_nested=True, dpi=216)
                plot_model(functional_model, to_file=''.join([plot_path, '.pdf']), show_shapes=True,
                           show_layer_names=True,
                           rankdir='TB', expand_nested=True)
                print('model analyzer ended with success, model saved in json, pdf and png formats\n')
            elif local_a_settings['use_efficientNetB2'] == 'True':
                # model_efficientNetB2 = models.load_model(local_a_settings['models_path'],
                #                                          '_7_layers_CNN__EfficientNetB2')
                # plot_model(model_efficientNetB2, to_file=''.join([plot_path, '.pdf']), show_shapes=True,
                #            show_layer_names=True, rankdir='TB', expand_nested=True)
                from contextlib import redirect_stdout
                with open(''.join([local_a_settings['models_path'], 'EfficientNetB2_summary.txt']), 'w') as f:
                    with redirect_stdout(f):
                        local_model.summary()
                    f.close()
                print('model analyzer ended with success, model EfficientNetB2 saved in json and txt formats\n')
                return True
        except Exception as e1:
            print('Error reading or saving model structure to pdf or png')
            print(e1)
            logger.error(str(e1), exc_info=True)
            return False
        return True
