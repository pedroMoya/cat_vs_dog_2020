# build CNN model
import os
import sys
import logging
import logging.handlers as handlers
import json
import numpy as np
import jpegio as jio

# open local settings
with open('./settings.json') as local_json_file:
    local_submodule_settings = json.loads(local_json_file.read())
    local_json_file.close()

# log setup
current_script_name = os.path.basename(__file__).split('.')[0]
log_path_filename = ''.join([local_submodule_settings['log_path'], current_script_name, '.log'])
logging.basicConfig(filename=log_path_filename, level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)
logHandler = handlers.RotatingFileHandler(log_path_filename, maxBytes=10485760, backupCount=5)
logger.addHandler(logHandler)

# load custom libraries
sys.path.insert(1, local_submodule_settings['custom_library_path'])

# class definitions


class quality_factor:

    def detect(self, local_img_path):
        local_quality_factor = 'other'
        try:
            # analyze
            jpeg = jio.read(local_img_path)
            quant_tables = jpeg.quant_tables
            # print(local_img_path.split('/')[-1])
            if quant_tables[0][0, 0] == 2:
                local_quality_factor = '2'
            elif quant_tables[0][0, 0] == 3:
                local_quality_factor = '1'
            elif quant_tables[0][0, 0] == 8:
                local_quality_factor = '0'
            else:
                print('error at estimating quality factor')
        except Exception as e:
            print('controlled error in quality_factor submodule')
            print(e)
            logger.error(str(e), exc_info=True)
        return local_quality_factor
