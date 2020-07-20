# the classifier derived from alternative training
import os
import sys
import fnmatch
import logging
from logging import handlers
import numpy as np
import json
import pandas as pd
from PIL import Image
import tensorflow as tf
import cv2
import itertools as it

with open('./settings.json') as local_json_file:
    local_settings = json.loads(local_json_file.read())
    local_json_file.close()
sys.path.insert(1, local_settings['custom_library_path'])
from alternative_classifier import alternative_classifier

# log setup
current_script_name = os.path.basename(__file__).split('.')[0]
log_path_filename = ''.join([local_settings['log_path'], current_script_name, '.log'])
logging.basicConfig(filename=log_path_filename, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logHandler = handlers.RotatingFileHandler(log_path_filename, maxBytes=10485760, backupCount=5)
logger.addHandler(logHandler)


class alternative_evaluation():

    def run(self, local_aeval_settings):
        try:
            type_of_model = '_EfficientNetB2'
            alt_classifier = tf.keras.applications.EfficientNetB2(include_top=True, weights='imagenet',
                                                                   input_tensor=None, input_shape=None,
                                                                   pooling=None, classes=1000,
                                                                   classifier_activation='softmax')
            local_model_evaluation_folder = ''.join([local_aeval_settings['models_evaluation_path'], 
                                                     'images_for_evaluation/'])
            nof_evaluation_samples_by_group = local_aeval_settings['nof_evaluation_samples_by_group']
            nof_methods = local_aeval_settings['nof_methods']
            nof_K_fold_groups = local_aeval_settings['nof_K_fold_groups']
            alt_model_classifier_instance = alternative_classifier()
            results = []
            ground_truth = 0
            for method, group in it.product(range(nof_methods), range(nof_K_fold_groups)):
                local_path = ''.join([local_model_evaluation_folder, 'method_', str(method), 
                                     '_group_', str(group), '/'])
                files = []
                if method > 0:
                    ground_truth = 1
                files = os.listdir(local_path)
                for image_file in files:
                    filepath = os.path.join(local_path, image_file)
                    local_image = Image.open(filepath)
                    local_prediction = alt_model_classifier_instance.think(local_image, local_aeval_settings,
                                                                           alt_classifier)
                    results.append([ground_truth, local_prediction])
            results_df = pd.DataFrame(results)
            print(np.shape(results_df))
            column_names = ['label', 'pred']
            results_df.to_csv(''.join([local_aeval_settings['models_evaluation_path'],
                                       'alternative_model_evaluation.csv']), index=False, header=column_names)
        except Exception as e3:
            print('error in alternative_model evaluation submodule')
            print(e3)
            logger.error(str(e3), exc_info=True)
            return False
        return True
