# match the same images and creates evaluation folders with K_FOLD_group (stored in 4_TRAIN_DATA_PATH)
import os
import sys
import shutil
import random
import itertools as it
import logging
import logging.handlers as handlers
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

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

# keras session/random seed reset/fix
np.random.seed(1)
random.seed(42)
tf.random.set_seed(2)

# functions definitions


# classes definitions
class k_fold_builder:

    def assign(self, local_id, local_nof_groups):
        try:
            # if more complex disaggregation will be need, here there are the submodule
            local_group = np.remainder(local_id, local_nof_groups)
        except Exception as e1:
            print('Error at matched_images K_fold_group dataset creation submodule')
            print(e1)
            logger.error(str(e1), exc_info=True)
            return False
        return local_group
