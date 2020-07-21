# open clean data, conform data structures
# training and saving models

# importing python libraries and opening settings
try:
    import os
    import sys
    import logging
    import logging.handlers as handlers
    import json
    import datetime
    import PIL
    import cv2
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    tf.keras.backend.set_floatx('float32')
    from tensorflow.keras import layers, models
    from tensorflow.keras import backend as kb
    from tensorflow.keras import preprocessing
    from tensorflow.keras import regularizers
    from tensorflow.keras import optimizers
    from tensorflow.keras import losses
    from tensorflow.keras import metrics
    from tensorflow.keras import callbacks as cb

    # open local settings
    with open('./settings.json') as local_json_file:
        local_settings = json.loads(local_json_file.read())
        local_json_file.close()

    # import custom libraries
    sys.path.insert(1, local_settings['custom_library_path'])
    from custom_model_builder import model_classifier_
    from custom_normalizer import custom_image_normalizer
    from lsbit_custom_normalizer import lsbit_custom_image_normalizer

except Exception as ee1:
    print('Error importing libraries or opening settings (train module)')
    print(ee1)

# log setup
current_script_name = os.path.basename(__file__).split('.')[0]
log_path_filename = ''.join([local_settings['log_path'], current_script_name, '.log'])
logging.basicConfig(filename=log_path_filename, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logHandler = handlers.RotatingFileHandler(log_path_filename, maxBytes=10485760, backupCount=5)
logger.addHandler(logHandler)


# classes definitions


# functions definitions


def image_normalizer(local_image_rgb):
    custom_image_normalizer_instance = custom_image_normalizer()
    return custom_image_normalizer_instance.normalize(local_image_rgb)


def train():
    print('\n~train_model module~')

    # keras,tf session/random seed reset/fix
    kb.clear_session()
    tf.compat.v1.reset_default_graph()
    np.random.seed(11)
    tf.random.set_seed(2)
    # tf.compat.v1.disable_eager_execution()

    # load model hyperparameters
    try:
        with open('./settings.json') as local_r_json_file:
            local_script_settings = json.loads(local_r_json_file.read())
            local_r_json_file.close()
        if local_script_settings['metaheuristic_optimization'] == "True":
            print('changing settings control to metaheuristic optimization')
            with open(''.join(
                    [local_script_settings['metaheuristics_path'], 'organic_settings.json'])) as local_r_json_file:
                local_script_settings = json.loads(local_r_json_file.read())
                local_r_json_file.close()

        # opening hyperparameters
        with open(''.join([local_script_settings['hyperparameters_path'], 'model_hyperparameters.json'])) \
                as local_r_json_file:
            model_hyperparameters = json.loads(local_r_json_file.read())
            local_r_json_file.close()

        if local_script_settings['data_cleaning_done'] == 'False':
            print('data was not cleaning')
            print('first prepare_data module have to run')
            return False

    except Exception as e1:
        print('Error loading model hyperparameters (train module)')
        print(e1)
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' error at loading model hyperparameters']))
        logger.error(str(e1), exc_info=True)

    # register model hyperparameters settings in log
    logging.info("\nexecuting train module program..\ncurrent models hyperparameters settings:%s",
                 ''.join(['\n', str(model_hyperparameters).replace(',', '\n')]))
    print('-current models hyperparameters registered in log')

    # starting training
    try:
        print('\n~train_model module~')
        # check settings for previous training and then repeat or not this phase
        if local_script_settings['training_done'] == "True":
            print('training of neural_network previously done')
            if local_script_settings['repeat_training'] == "True":
                print('repeating training')
            else:
                print("settings indicates don't repeat training")
                return True
        else:
            print('model training start')

        # model training hyperparameters
        nof_groups = local_script_settings['nof_K_fold_groups']
        input_shape_y = model_hyperparameters['input_shape_y']
        input_shape_x = model_hyperparameters['input_shape_x']
        epochs = model_hyperparameters['epochs']
        batch_size = model_hyperparameters['batch_size']
        workers = model_hyperparameters['workers']
        validation_split = model_hyperparameters['validation_split']
        early_stopping_patience = model_hyperparameters['early_stopping_patience']
        reduce_lr_on_plateau_factor = model_hyperparameters['ReduceLROnPlateau_factor']
        reduce_lr_on_plateau_patience = model_hyperparameters['ReduceLROnPlateau_patience']
        reduce_lr_on_plateau_min_lr = model_hyperparameters['ReduceLROnPlateau_min_lr']
        validation_freq = model_hyperparameters['validation_freq']
        training_set_folder = model_hyperparameters['training_set_folder']
        validation_set_folder = model_hyperparameters['validation_set_folder']

        # load raw_data and cleaned_data
        column_names = ['id_number', 'class', 'group', 'filename', 'filepath']
        x_col = 'filepath'
        y_col = 'class'
        metadata_train_images = \
            pd.read_csv(''.join([local_script_settings['train_data_path'], 'training_metadata.csv']),
                        dtype=str, names=column_names, header=None)

        train_datagen = preprocessing.image.ImageDataGenerator(rescale=None,
                                                               horizontal_flip=True,
                                                               width_shift_range=0.1,
                                                               height_shift_range=0.1,
                                                               rotation_range=15,
                                                               shear_range=0.1,
                                                               zoom_range=0.2,
                                                               validation_split=validation_split)
        train_generator = train_datagen.flow_from_dataframe(dataframe=metadata_train_images,
                                                            directory=None,
                                                            x_col=x_col,
                                                            y_col=y_col,
                                                            target_size=(input_shape_y, input_shape_x),
                                                            batch_size=batch_size,
                                                            class_mode='categorical',
                                                            color_mode='rgb',
                                                            shuffle=True,
                                                            subset='training')
        print('classes and indices of train_generator')
        print(train_generator.class_indices)
        validation_datagen = \
            preprocessing.image.ImageDataGenerator(rescale=None,
                                                   validation_split=validation_split)
        validation_generator = validation_datagen.flow_from_dataframe(dataframe=metadata_train_images,
                                                                      directory=None,
                                                                      x_col=x_col,
                                                                      y_col=y_col,
                                                                      target_size=(input_shape_y, input_shape_x),
                                                                      batch_size=batch_size,
                                                                      class_mode='categorical',
                                                                      color_mode='rgb',
                                                                      shuffle=True,
                                                                      subset='validation')
        print('classes and indices of validation_generator')
        print(validation_generator.class_indices)

        # build, compile and save model
        model_name = model_hyperparameters['current_model_name']
        classifier_constructor = model_classifier_()
        classifier = classifier_constructor.build_and_compile(model_name, local_script_settings,
                                                              model_hyperparameters)
        if isinstance(classifier, int):
            print('build and compile not return a model (may be a correct behavior if alternative training is set)')
            if local_script_settings['alternative_training'] == 'True':
                return True
            else:
                print('error, please review consistency of settings and code if necessary')
                return False

        # define callbacks, checkpoints namepaths
        model_weights = ''.join([local_settings['checkpoints_path'],
                                 'check_point_', model_name, "_loss_-{loss:.4f}-.hdf5"])
        callback1 = cb.EarlyStopping(monitor='loss', patience=early_stopping_patience)
        callback2 = [cb.ModelCheckpoint(model_weights, monitor='loss', verbose=1,
                                        save_best_only=True, mode='min')]
        callback3 = cb.ReduceLROnPlateau(monitor='loss', factor=reduce_lr_on_plateau_factor,
                                         patience=reduce_lr_on_plateau_patience,
                                         min_lr=reduce_lr_on_plateau_min_lr)
        callbacks = [callback1, callback2, callback3]

        # class weights as imbalanced dataset
        if model_hyperparameters['class_weights'] != "None":
            class_weight = {0: 0.666, 1: 0.333}
        else:
            class_weight = None

        # training model
        model_train_history = classifier.fit(train_generator, epochs=epochs, batch_size=batch_size,
                                             steps_per_epoch=train_generator.samples // batch_size,
                                             callbacks=callbacks, shuffle=True, workers=workers,
                                             class_weight=class_weight,
                                             validation_data=validation_generator,
                                             validation_freq=validation_freq,
                                             validation_steps=validation_generator.samples // batch_size,
                                             use_multiprocessing=False)

        # save weights (model was saved previously at model build and compile in h5 and json formats)
        if local_script_settings['use_efficientNetB2'] == "True":
            type_of_model = '_EfficientNetB2'
        elif local_script_settings['use_efficientNetB2'] == "True":
            type_of_model = '_custom'
        else:
            print('type of model not understood')
            return False
        date = datetime.date.today()
        classifier.save_weights(''.join([local_script_settings['models_path'], model_name, '_', type_of_model,
                                         '_', str(date), '_weights.h5']))

        # save in tf (saveModel) format
        classifier.save(''.join([local_settings['models_path'], model_name, type_of_model, '_trained_/']),
                        save_format='tf')

        # closing train module
        print('full training of each group module ended')
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' correct model training, correct saving of model and weights']))
        local_script_settings['training_done'] = "True"
        if local_script_settings['metaheuristic_optimization'] == "False":
            with open('./settings.json', 'w', encoding='utf-8') as local_wr_json_file:
                json.dump(local_script_settings, local_wr_json_file, ensure_ascii=False, indent=2)
                local_wr_json_file.close()
        elif local_script_settings['metaheuristic_optimization'] == "True":
            with open(''.join([local_script_settings['metaheuristics_path'],
                               'organic_settings.json']), 'w', encoding='utf-8') as local_wr_json_file:
                json.dump(local_script_settings, local_wr_json_file, ensure_ascii=False, indent=2)
                local_wr_json_file.close()
                # metaheuristic_train = tuning_metaheuristic()
                # metaheuristic_hyperparameters = metaheuristic_train.stochastic_brain(local_script_settings)
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' settings modified and saved']))
    except Exception as e1:
        print('Error training model')
        print(e1)
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' model training error']))
        logger.error(str(e1), exc_info=True)
        return False
    return True
