# build CNN model
import os
import sys
import datetime
import logging
import logging.handlers as handlers
import json
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
tf.keras.backend.set_floatx('float32')
from tensorflow.keras import layers
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

# load custom libraries
sys.path.insert(1, local_submodule_settings['custom_library_path'])
from model_analyzer import model_structure


# class definitions


class customized_metrics_bacc(metrics.Metric):
    @tf.function
    def call(self, y_true_local, y_pred_local, fp=metrics.FalsePositives(), fn=metrics.FalseNegatives(),
             tp=metrics.TruePositives(), tn=metrics.TrueNegatives()):
        return 0.5 * tn / (tn + fn) + (tp / (tp + fp))


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


class model_classifier_:

    def build_and_compile(self, local_model_name, local_settings, local_hyperparameters):
        try:
            # keras,tf session/random seed reset/fix
            kb.clear_session()
            tf.compat.v1.reset_default_graph()
            np.random.seed(11)
            tf.random.set_seed(2)

            # load hyperparameters
            units_layer_1 = local_hyperparameters['units_layer_1']
            units_layer_2 = local_hyperparameters['units_layer_2']
            units_layer_3 = local_hyperparameters['units_layer_3']
            units_layer_4 = local_hyperparameters['units_layer_4']
            units_last_layer_con2d_efficientnetb2 = local_hyperparameters['units_layer_last_conv2d_efficientnetb2']
            units_final_layer = local_hyperparameters['units_final_layer']
            activation_1 = local_hyperparameters['activation_1']
            activation_2 = local_hyperparameters['activation_2']
            activation_3 = local_hyperparameters['activation_3']
            activation_4 = local_hyperparameters['activation_4']
            activation_last_layer_con2d_efficientnetb2 = \
                local_hyperparameters['activation_last_layer_con2d_efficientnetb2']
            activation_final_layer = local_hyperparameters['activation_final_layer']
            dropout_layer_1 = local_hyperparameters['dropout_layer_1']
            dropout_layer_2 = local_hyperparameters['dropout_layer_2']
            dropout_layer_3 = local_hyperparameters['dropout_layer_3']
            dropout_layer_4 = local_hyperparameters['dropout_layer_4']
            dropout_dense_layer_4 = local_hyperparameters['dropout_dense_layer_4']
            input_shape_y = local_hyperparameters['input_shape_y']
            input_shape_x = local_hyperparameters['input_shape_x']
            nof_channels = local_hyperparameters['nof_channels']
            stride_y_1 = local_hyperparameters['stride_y_1']
            stride_x_1 = local_hyperparameters['stride_x_1']
            kernel_size_y_1 = local_hyperparameters['kernel_size_y_1']
            kernel_size_x_1 = local_hyperparameters['kernel_size_x_1']
            kernel_size_y_2 = local_hyperparameters['kernel_size_y_2']
            kernel_size_x_2 = local_hyperparameters['kernel_size_x_2']
            kernel_size_y_3 = local_hyperparameters['kernel_size_y_3']
            kernel_size_x_3 = local_hyperparameters['kernel_size_x_3']
            kernel_size_y_4 = local_hyperparameters['kernel_size_y_4']
            kernel_size_x_4 = local_hyperparameters['kernel_size_x_4']
            pool_size_y_1 = local_hyperparameters['pool_size_y_1']
            pool_size_x_1 = local_hyperparameters['pool_size_x_1']
            pool_size_y_2 = local_hyperparameters['pool_size_y_2']
            pool_size_x_2 = local_hyperparameters['pool_size_x_2']
            pool_size_y_3 = local_hyperparameters['pool_size_y_3']
            pool_size_x_3 = local_hyperparameters['pool_size_x_3']
            pool_size_y_4 = local_hyperparameters['pool_size_y_4']
            pool_size_x_4 = local_hyperparameters['pool_size_x_4']
            optimizer_function = local_hyperparameters['optimizer']
            optimizer_learning_rate = local_hyperparameters['learning_rate']
            if optimizer_function == 'adam':
                optimizer_function = optimizers.Adam(optimizer_learning_rate)
                optimizer_function = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer_function)
            elif optimizer_function == 'ftrl':
                optimizer_function = optimizers.Ftrl(optimizer_learning_rate)
            elif optimizer_function == 'sgd':
                optimizer_function = optimizers.SGD(optimizer_learning_rate)
            elif optimizer_function == 'rmsp':
                optimizer_function = optimizers.RMSprop(lr=2e-5)
            losses_list = []
            loss_1 = local_hyperparameters['loss_1']
            loss_2 = local_hyperparameters['loss_2']
            loss_3 = local_hyperparameters['loss_3']
            union_settings_losses = [loss_1, loss_2, loss_3]
            if 'CategoricalCrossentropy' in union_settings_losses:
                losses_list.append(losses.CategoricalCrossentropy())
            if 'BinaryCrossentropy' in union_settings_losses:
                losses_list.append(losses.BinaryCrossentropy())
            if 'CategoricalHinge' in union_settings_losses:
                losses_list.append(losses.CategoricalHinge())
            if 'LogCosh' in union_settings_losses:
                losses_list.append(losses.LogCosh)
            if 'customized_loss_function' in union_settings_losses:
                losses_list.append(customized_loss())
            if 'customized_loss_auc_roc' in union_settings_losses:
                losses_list.append(customized_loss_auc_roc())
            if "Huber" in union_settings_losses:
                losses_list.append(losses.Huber())
            metrics_list = []
            metric1 = local_hyperparameters['metrics1']
            metric2 = local_hyperparameters['metrics2']
            union_settings_metrics = [metric1, metric2]
            if 'auc_roc' in union_settings_metrics:
                metrics_list.append(metrics.AUC())
            if 'CategoricalAccuracy' in union_settings_metrics:
                metrics_list.append(metrics.CategoricalAccuracy())
            if 'CategoricalHinge' in union_settings_metrics:
                metrics_list.append(metrics.CategoricalHinge())
            if 'BinaryAccuracy' in union_settings_metrics:
                metrics_list.append(metrics.BinaryAccuracy())
            if local_settings['use_efficientNetB2'] == 'False':
                type_of_model = '_custom'
                if local_hyperparameters['regularizers_l1_l2_1'] == 'True':
                    l1_1 = local_hyperparameters['l1_1']
                    l2_1 = local_hyperparameters['l2_1']
                    activation_regularizer_1 = regularizers.l1_l2(l1=l1_1, l2=l2_1)
                else:
                    activation_regularizer_1 = None
                if local_hyperparameters['regularizers_l1_l2_2'] == 'True':
                    l1_2 = local_hyperparameters['l1_2']
                    l2_2 = local_hyperparameters['l2_2']
                    activation_regularizer_2 = regularizers.l1_l2(l1=l1_2, l2=l2_2)
                else:
                    activation_regularizer_2 = None
                if local_hyperparameters['regularizers_l1_l2_3'] == 'True':
                    l1_3 = local_hyperparameters['l1_3']
                    l2_3 = local_hyperparameters['l2_3']
                    activation_regularizer_3 = regularizers.l1_l2(l1=l1_3, l2=l2_3)
                else:
                    activation_regularizer_3 = None
                if local_hyperparameters['regularizers_l1_l2_4'] == 'True':
                    l1_4 = local_hyperparameters['l1_4']
                    l2_4 = local_hyperparameters['l2_4']
                    activation_regularizer_4 = regularizers.l1_l2(l1=l1_4, l2=l2_4)
                else:
                    activation_regularizer_4 = None
                if local_hyperparameters['regularizers_l1_l2_dense_4'] == 'True':
                    l1_dense_4 = local_hyperparameters['l1_dense_4']
                    l2_dense_4 = local_hyperparameters['l2_dense_4']
                    activation_regularizer_dense_layer_4 = regularizers.l1_l2(l1=l1_dense_4, l2=l2_dense_4)
                else:
                    activation_regularizer_dense_layer_4 = None

                # building model
                classifier_ = tf.keras.models.Sequential()
                # first layer
                classifier_.add(layers.Input(shape=(input_shape_y, input_shape_x, nof_channels)))
                # classifier_.add(layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
                classifier_.add(layers.Conv2D(units_layer_1, kernel_size=(kernel_size_y_1, kernel_size_x_1),
                                              strides=(stride_y_1, stride_x_1),
                                              activity_regularizer=activation_regularizer_1,
                                              activation=activation_1,
                                              padding='same',
                                              kernel_initializer=tf.keras.initializers.VarianceScaling(
                                                  scale=2., mode='fan_out', distribution='truncated_normal')))
                classifier_.add(layers.BatchNormalization(axis=-1))
                classifier_.add(layers.Activation(tf.keras.activations.swish))
                classifier_.add(layers.GlobalAveragePooling2D())
                classifier_.add(layers.Dropout(dropout_layer_1))
                # LAYER 1.5
                classifier_.add(layers.Conv2D(units_layer_1, kernel_size=(kernel_size_y_1, kernel_size_x_1),
                                              input_shape=(input_shape_y, input_shape_x, nof_channels),
                                              strides=(stride_y_1, stride_x_1),
                                              activity_regularizer=activation_regularizer_1,
                                              activation=activation_1,
                                              padding='same',
                                              kernel_initializer=tf.keras.initializers.VarianceScaling(
                                                  scale=2., mode='fan_out', distribution='truncated_normal')))
                classifier_.add(layers.BatchNormalization(axis=-1))
                classifier_.add(layers.Activation(tf.keras.activations.swish))
                classifier_.add(layers.GlobalAveragePooling2D())
                classifier_.add(layers.Dropout(dropout_layer_1))
                # second layer
                classifier_.add(layers.Conv2D(units_layer_2, kernel_size=(kernel_size_y_2, kernel_size_x_2),
                                              activity_regularizer=activation_regularizer_2,
                                              activation=activation_2,
                                              padding='same',
                                              kernel_initializer=tf.keras.initializers.VarianceScaling(
                                                  scale=2., mode='fan_out', distribution='truncated_normal')))
                classifier_.add(layers.BatchNormalization(axis=-1))
                classifier_.add(layers.Activation(tf.keras.activations.swish))
                classifier_.add(layers.GlobalAveragePooling2D())
                classifier_.add(layers.Dropout(dropout_layer_2))
                # LAYER 2.5
                classifier_.add(layers.Conv2D(units_layer_2, kernel_size=(kernel_size_y_2, kernel_size_x_2),
                                              activity_regularizer=activation_regularizer_2,
                                              activation=activation_2,
                                              padding='same',
                                              kernel_initializer=tf.keras.initializers.VarianceScaling(
                                                  scale=2., mode='fan_out', distribution='truncated_normal')))
                classifier_.add(layers.BatchNormalization(axis=-1))
                classifier_.add(layers.Activation(tf.keras.activations.swish))
                classifier_.add(layers.GlobalAveragePooling2D())
                classifier_.add(layers.Dropout(dropout_layer_2))
                # third layer
                classifier_.add(layers.Conv2D(units_layer_3,
                                              kernel_size=(kernel_size_y_3, kernel_size_x_3),
                                              activity_regularizer=activation_regularizer_3,
                                              activation=activation_3,
                                              padding='same',
                                              kernel_initializer=tf.keras.initializers.VarianceScaling(
                                                  scale=2., mode='fan_out', distribution='truncated_normal')))
                classifier_.add(layers.BatchNormalization(axis=-1))
                classifier_.add(layers.Activation(tf.keras.activations.swish))
                classifier_.add(layers.GlobalAveragePooling2D())
                classifier_.add(layers.Dropout(dropout_layer_3))
                # LAYER 3.5
                classifier_.add(layers.Conv2D(units_layer_3,
                                              kernel_size=(kernel_size_y_3, kernel_size_x_3),
                                              activity_regularizer=activation_regularizer_3,
                                              activation=activation_3,
                                              padding='same',
                                              kernel_initializer=tf.keras.initializers.VarianceScaling(
                                                  scale=2., mode='fan_out', distribution='truncated_normal')))
                classifier_.add(layers.BatchNormalization(axis=-1))
                classifier_.add(layers.Activation(tf.keras.activations.swish))
                classifier_.add(layers.GlobalAveragePooling2D())
                classifier_.add(layers.Dropout(dropout_layer_3))
                # fourth layer
                classifier_.add(layers.Conv2D(units_layer_4,
                                              kernel_size=(kernel_size_y_4, kernel_size_x_4),
                                              activity_regularizer=activation_regularizer_4,
                                              activation=activation_4,
                                              padding='same',
                                              kernel_initializer=tf.keras.initializers.VarianceScaling(
                                                  scale=2., mode='fan_out', distribution='truncated_normal')))
                classifier_.add(layers.BatchNormalization(axis=-1))
                classifier_.add(layers.Activation(tf.keras.activations.swish))
                classifier_.add(layers.GlobalAveragePooling2D())
                classifier_.add(layers.Dropout(dropout_layer_4))
                # Full connection and final layer
                classifier_.add(layers.Dense(units=units_final_layer, activation=activation_final_layer))
                # Compile model
                classifier_.compile(optimizer=optimizer_function, loss=losses_list, metrics=metrics_list)

            elif local_settings['use_efficientNetB2'] == 'True':
                type_of_model = '_EfficientNetB2'
                # pretrained_weights = ''.join([local_settings['models_path'],
                #                               local_hyperparameters['weights_for_training_efficientnetb2']])
                classifier_ = tf.keras.applications.EfficientNetB2(include_top=False, weights='imagenet',
                                                                   input_tensor=None, input_shape=None,
                                                                   pooling=None,
                                                                   classifier_activation='softmax')
                classifier_.trainable = True
                # for layer in classifier_.layers:
                #     layer.trainable = False
                #     if 'excite' in layer.name:
                #         layer.trainable = True
                #     if 'top_conv' in layer.name:
                #         layer.trainable = False
                #     if 'block7b_project_conv' in layer.name:
                #         layer.trainable = False

                if local_settings['nof_methods'] == 2:
                    # if two classes, and imbalanced, bias_initializer = log(pos/neg)
                    bias_initializer = tf.keras.initializers.Constant(local_hyperparameters['bias_initializer'])
                else:
                    bias_initializer = tf.keras.initializers.Constant(0)

                effnb2_model = models.Sequential()
                effnb2_model.add(classifier_)
                effnb2_model.add(layers.GlobalMaxPool2D())
                effnb2_model.add(layers.Dropout(dropout_dense_layer_4))
                effnb2_model.add(layers.Dense(units_final_layer, activation=activation_final_layer,
                                 kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.333333333,
                                                                                          mode='fan_out',
                                                                                          distribution='uniform'),
                                              bias_initializer=bias_initializer))
                effnb2_model.build(input_shape=(input_shape_y, input_shape_x, nof_channels))
                effnb2_model.compile(optimizer=optimizer_function, loss=losses_list, metrics=metrics_list)
                classifier_ = effnb2_model

            else:
                print('model to use is not defined')
                return False

            # Summary of model
            classifier_.summary()

            # save_model
            classifier_json = classifier_.to_json()
            with open(''.join([local_settings['models_path'], local_model_name, type_of_model,
                               '_classifier_.json']), 'w') \
                    as json_file:
                json_file.write(classifier_json)
                json_file.close()
            classifier_.save(''.join([local_settings['models_path'], local_model_name, type_of_model,
                                      '_classifier_.h5']))
            classifier_.save(''.join([local_settings['models_path'], local_model_name, type_of_model,
                                      '/']), save_format='tf')
            print('model architecture saved')

            # output png and pdf with model, additionally saves a json file model_name_analyzed.json
            if local_settings['model_analyzer'] == 'True':
                model_architecture = model_structure()
                model_architecture_review = model_architecture.analize(''.join([local_model_name, type_of_model,
                                                                                '_classifier_.h5']),
                                                                       local_settings, local_hyperparameters)
        except Exception as e:
            print('error in build or compile of customized model')
            print(e)
            classifier_ = None
            logger.error(str(e), exc_info=True)
        return classifier_
