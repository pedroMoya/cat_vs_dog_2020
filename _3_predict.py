# read input data, execute model(s) and save forecasts

# importing python libraries and opening settings
try:
    import os
    import sys
    import logging
    import logging.handlers as handlers
    import json
    import datetime
    import numpy as np
    import pandas as pd
    import itertools as it
    import tensorflow as tf
    from tensorflow.keras import backend as kb
    from tensorflow.keras import losses, models
    from tensorflow.keras.metrics import mean_absolute_percentage_error
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    tf.keras.backend.set_floatx('float32')

    with open('./settings.json') as local_json_file:
        local_script_settings = json.loads(local_json_file.read())
        local_json_file.close()

    sys.path.insert(1, local_script_settings['custom_library_path'])

    if local_script_settings['metaheuristic_optimization'] == "True":
        with open(''.join(
                [local_script_settings['metaheuristics_path'], 'organic_settings.json'])) as local_json_file:
            local_script_settings = json.loads(local_json_file.read())
            local_json_file.close()
        # metaheuristic_predict = tuning_metaheuristic()
except Exception as ee1:
    print('Error importing libraries or opening settings (predict module)')
    print(ee1)

# log setup
current_script_name = os.path.basename(__file__).split('.')[0]
log_path_filename = ''.join([local_script_settings['log_path'], current_script_name, '.log'])
logging.basicConfig(filename=log_path_filename, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logHandler = handlers.RotatingFileHandler(log_path_filename, maxBytes=10485760, backupCount=5)
logger.addHandler(logHandler)

# keras session/random seed reset/fix
kb.clear_session()
np.random.seed(1)
tf.random.set_seed(2)


# classes definitions


# functions definitions


def general_mean_scaler(local_array):
    if len(local_array) == 0:
        return "argument length 0"
    mean_local_array = np.mean(local_array, axis=1)
    mean_scaling = np.divide(local_array, 1 + mean_local_array)
    return mean_scaling, mean_local_array


def window_based_normalizer(local_window_array):
    if len(local_window_array) == 0:
        return "argument length 0"
    mean_local_array = np.mean(local_window_array, axis=1)
    window_based_normalized_array = np.add(local_window_array, -mean_local_array)
    return window_based_normalized_array, mean_local_array


def general_mean_rescaler(local_array, local_complete_array_unit_mean, local_forecast_horizon):
    if len(local_array) == 0:
        return "argument length 0"
    local_array = local_array.clip(0)
    local_complete_array_unit_mean = np.array([local_complete_array_unit_mean, ] * local_forecast_horizon).transpose()
    mean_rescaling = np.multiply(local_array, 1 + local_complete_array_unit_mean)
    return mean_rescaling


def window_based_denormalizer(local_window_array, local_last_window_mean, local_forecast_horizon):
    if len(local_window_array) == 0:
        return "argument length 0"
    local_last_window_mean = np.array([local_last_window_mean, ] * local_forecast_horizon).transpose()
    window_based_denormalized_array = np.add(local_window_array, local_last_window_mean)
    return window_based_denormalized_array


def predict():
    try:
        print('\n~predict module~')

        # open predict settings
        with open(''.join([local_script_settings['test_data_path'], 'forecast_settings.json'])) as local_f_json_file:
            forecast_settings = json.loads(local_f_json_file.read())
            local_f_json_file.close()

        # load clean data

        # extract data and check  dimensions

        # load model

        # make predictions

        # save predictions
        # np.savetxt(''.join([local_script_settings['others_outputs_path'], 'predictions_.csv']),
        #            predictions, fmt='%10.15f', delimiter=',', newline='\n')
        print('predictions saved to file')

        # saving consolidated submission
        submission_template = np.genfromtxt(''.join([local_script_settings['raw_data_path'], 'sample_submission.csv']),
                                            delimiter=',', dtype=None, encoding=None)

        # fill with predictions


        # save submission
        # pd.DataFrame(submission).to_csv(''.join([local_script_settings['submission_path'], 'submission.csv']),
        #                                 index=False, header=None)
        # np.savetxt(''.join([local_script_settings['others_outputs_path'],
        #                     'point_forecast_ss_and_or_nn_models_applied_.csv']),
        #            all_forecasts, fmt='%10.15f', delimiter=',', newline='\n')
        print('predictions saved, submission file built and stored')
        print("predictions process ended successfully")
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' correct prediction process']))
    except Exception as e1:
        print('Error in predict module')
        print(e1)
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' predict module error']))
        logger.error(str(e1), exc_info=True)
        return False
    return True
