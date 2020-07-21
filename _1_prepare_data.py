# preparing data (cleaning raw data, aggregating and saving to file)

# importing python libraries and opening settings
import os
import sys
import shutil
import logging
import logging.handlers as handlers
import json
import datetime
import numpy as np
import pandas as pd
import itertools as it
import tensorflow as tf
from tensorflow.keras import preprocessing

# open local settings and change local_scrip_settings if metaheuristic equals True
with open('./settings.json') as local_json_file:
    local_script_settings = json.loads(local_json_file.read())
    local_json_file.close()

# import custom libraries
sys.path.insert(1, local_script_settings['custom_library_path'])
from k_fold_data_creator import k_fold_builder

if local_script_settings['metaheuristic_optimization'] == "True":
    with open(''.join([local_script_settings['metaheuristics_path'],
                       'organic_settings.json'])) as local_json_file:
        local_script_settings = json.loads(local_json_file.read())
        local_json_file.close()

# log setup
current_script_name = os.path.basename(__file__).split('.')[0]
log_path_filename = ''.join([local_script_settings['log_path'], current_script_name, '.log'])
logging.basicConfig(filename=log_path_filename, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logHandler = handlers.RotatingFileHandler(log_path_filename, maxBytes=10485760, backupCount=5)
logger.addHandler(logHandler)
logger.info('_prepare_data module start')

# Random seed fixed
np.random.seed(1)

# functions definitions


def prepare():
    print('\n~prepare_data module~')
    # check if clean is done
    if local_script_settings['data_cleaning_done'] == "True":
        print('datasets already cleaned, based in settings info')
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' raw datasets already cleaned']))
        if local_script_settings['repeat_data_cleaning'] == "False":
            print('skipping prepare_data cleaning, as settings indicates')
            return True
        else:
            print('repeating data cleaning again')
            logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                                 ' cleaning raw datasets']))

    # pre-processing core
    try:
        # define filepaths
        raw_data_path = local_script_settings['raw_data_path']
        label_0_raw_data_path = ''.join([raw_data_path, local_script_settings['class_0_folder']])
        label_1_raw_data_path = ''.join([raw_data_path, local_script_settings['class_1_folder']])
        label_0_raw_data_evaluation_path = ''.join([raw_data_path, local_script_settings['class_0_evaluation_folder']])
        label_1_raw_data_evaluation_path = ''.join([raw_data_path, local_script_settings['class_1_evaluation_folder']])

        # extract files
        if not os.path.isfile(''.join([local_script_settings['raw_data_path'], 'images_localization.txt'])):
            images_label_0 = [''.join([label_0_raw_data_path, filename])
                              for filename in os.listdir(label_0_raw_data_path)]
            images_label_1 = [''.join([label_1_raw_data_path, filename])
                              for filename in os.listdir(label_1_raw_data_path)]
            evaluation_images_label_0 = [''.join([label_0_raw_data_evaluation_path, filename])
                                         for filename in os.listdir(label_0_raw_data_evaluation_path)]
            evaluation_images_label_1 = [''.join([label_1_raw_data_evaluation_path, filename])
                                         for filename in os.listdir(label_1_raw_data_evaluation_path)]

            images_loc = ','.join(images_label_0 + images_label_1)
            evaluation_images_loc = ','.join(evaluation_images_label_0 + evaluation_images_label_1)

            # save
            with open(''.join([local_script_settings['raw_data_path'], 'images_localization.txt']), 'w') as f:
                f.write(images_loc)
                f.close()
            images_loc = images_loc.split(',')
            with open(''.join([local_script_settings['raw_data_path'], 'evaluation_images_localization.txt']), 'w') as f:
                f.write(evaluation_images_loc)
                f.close()
            evaluation_images_loc = evaluation_images_loc.split(',')
        else:
            with open(''.join([local_script_settings['raw_data_path'], 'images_localization.txt'])) as f:
                chain = f.read()
                images_loc = chain.split(',')
                f.close()
            with open(''.join([local_script_settings['raw_data_path'], 'evaluation_images_localization.txt'])) as f:
                chain = f.read()
                evaluation_images_loc = chain.split(',')
                f.close()
        nof_images = len(images_loc)
        eval_nof_images = len(evaluation_images_loc)
        print('total jpg images found for training:', nof_images)
        print('total jpg images found for evaluation:', eval_nof_images)

        # open raw_data and disaggregation
        # format of training_metadata: [id_number, label, quality_factor, group, filename, filepath]
        id_number = 0
        nof_groups = local_script_settings['nof_K_fold_groups']
        training_metadata = []
        print('first pre-processing step: disaggregation')
        if local_script_settings['disaggregation_done'] == "False":
            # train dataset
            for image_path in images_loc:
                filename = image_path.split('/')[-1]
                # train_data_path_template = local_script_settings['raw_data_path']

                # K fold disaggregation
                k_fold_instance = k_fold_builder()
                group = np.int(k_fold_instance.assign(id_number, nof_groups))

                # detecting the label by folder or filename
                if 'cat' in image_path:
                    label = 'cat'
                    # train_data_path = ''.join([train_data_path_template, 'label_0_'])
                elif 'dog' in image_path:
                    label= 'dog'
                    # train_data_path = ''.join([train_data_path_template, 'label_1_'])
                else:
                    print('label not understood')
                    return False

                train_data_path_filename = image_path
                training_metadata.append([id_number, label, group, filename, train_data_path_filename])
                id_number += 1

            # validation dataset
            id_number = 0
            evaluation_metadata = []
            for image_path in evaluation_images_loc:
                filename = image_path.split('/')[-1]

                # detecting the label by folder or filename
                if 'cat' in image_path:
                    label = 'cat'
                elif 'dog' in image_path:
                    label = 'dog'
                else:
                    print('label not understood')
                    return False

                evaluation_data_path_filename = image_path
                evaluation_metadata.append([id_number, label, filename, evaluation_data_path_filename])
                id_number += 1

            # save clean metadata source for use in subsequent training
            training_metadata_df = pd.DataFrame(training_metadata)
            training_metadata_df.to_csv(''.join([local_script_settings['clean_data_path'],
                                                 'training_metadata.csv']), index=False, header=None)
            training_metadata_df.to_csv(''.join([local_script_settings['train_data_path'],
                                                 'training_metadata.csv']), index=False, header=None)
            evaluation_metadata_df = pd.DataFrame(evaluation_metadata)
            evaluation_metadata_df.to_csv(''.join([local_script_settings['models_evaluation_path'],
                                                   'evaluation_metadata.csv']), index=False, header=None)
            evaluation_metadata_df.to_csv(''.join([local_script_settings['models_evaluation_path'],
                                                   'evaluation_metadata.csv']), index=False, header=None)
            np.save(''.join([local_script_settings['clean_data_path'], 'training_metadata_np']),
                    training_metadata)
            np.save(''.join([local_script_settings['clean_data_path'], 'evaluation_metadata_np']),
                    evaluation_metadata)
            print('train and evaluation data -and their metadata- saved to file')
            logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                                 ' successful saved training data and correspondent metadata']))
            with open('./settings.json', 'w', encoding='utf-8') as local_wr_json_file:
                local_script_settings['disaggregation_done'] = "True"
                json.dump(local_script_settings, local_wr_json_file, ensure_ascii=False, indent=2)
                local_wr_json_file.close()
            print('data aggregation was done')
        elif local_script_settings['disaggregation_done'] == "True":
            print('data disaggregation was done previously')
        else:
            print('settings disaggregation not understood')
            return False

        # data general_mean based - scaling
        # this step is automatically done in train by ImageDataGenerator
        print('data scaling was correctly prepared')

        # data normalization based in moving window
        # this step is included as a pre-processing_function in ImageDataGenerator
        print('data normalization was also prepared as a pre-processing_function (on the fly)')

        # save clean metadata source for use in subsequent training
        # if local_script_settings['disaggregation_done'] == "False":
        #     training_metadata_df = pd.DataFrame(training_metadata)
        #     column_names = ['id_number', 'label', 'quality_factor', 'group', 'filename', 'filepath']
        #     training_metadata_df.to_csv(''.join([local_script_settings['clean_data_path'],
        #                                          'training_metadata.csv']), index=False, header=column_names)
        #     np.save(''.join([local_script_settings['clean_data_path'], 'training_metadata_np']),
        #             training_metadata)
        #     np.savetxt(''.join([local_script_settings['clean_data_path'], 'training_metadata_np_to.csv']),
        #                training_metadata, fmt='%10.15f', delimiter=',', newline='\n')
        #     print('train data -and their metadata- saved to file')
        #     logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
        #                          ' successful saved training data and correspondent metadata']))
    except Exception as e1:
        print('Error at pre-processing raw data')
        print(e1)
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' data pre-processing error']))
        logger.error(str(e1), exc_info=True)
        return False

    # save settings
    try:
        if local_script_settings['metaheuristic_optimization'] == "False":
            with open('./settings.json', 'w', encoding='utf-8') as local_wr_json_file:
                local_script_settings['data_cleaning_done'] = "True"
                json.dump(local_script_settings, local_wr_json_file, ensure_ascii=False, indent=2)
                local_wr_json_file.close()
        elif local_script_settings['metaheuristic_optimization'] == "True":
            with open(''.join([local_script_settings['metaheuristics_path'],
                               'organic_settings.json']), 'w', encoding='utf-8') as local_wr_json_file:
                local_script_settings['data_cleaning_done'] = "True"
                json.dump(local_script_settings, local_wr_json_file, ensure_ascii=False, indent=2)
                local_wr_json_file.close()
        logger.info(''.join(['\n', datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"),
                             ' settings modified and saved']))
        print("raw datasets cleaned, settings saved..")
    except Exception as e1:
        print('Error saving settings')
        print(e1)
        logger.error(str(e1), exc_info=True)

    # back to main code
    return True
