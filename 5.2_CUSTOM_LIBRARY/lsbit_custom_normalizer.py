# function that makes last pre-process step to images, preparing data for training the CNNs
# use vectorize and LSByte
import numpy as np
import itertools as it
import tensorflow as tf
import cv2


class lsbit_custom_image_normalizer():

    def normalize(self, local_image_rgb):
        local_image_rgb = cv2.cvtColor(local_image_rgb, cv2.COLOR_RGB2YCR_CB)
        channel_y = local_image_rgb[:, :, 0: 1]
        # channel_y_lsbit = np.bitwise_and(channel_y.astype(np.int32), 1)
        # * 255 if using EfficientNetB2
        channel_y_lsbit = np.bitwise_and(channel_y.astype(np.int32), 1) * 255
        local_image_rgb[:, :, 2: 3] = channel_y_lsbit
        local_image_rgb[:, :, 1: 2] = np.abs(np.add(local_image_rgb[:, :, 1: 2], channel_y_lsbit))
        local_image_rgb[:, :, 0: 1] = np.abs(np.add(local_image_rgb[:, :, 0: 1], channel_y_lsbit))
        # local_max = np.amax(channel_y, axis=(0, 1))
        # local_min = np.amin(channel_y, axis=(0, 1))
        # local_denom_diff = np.add(local_max, -local_min)
        # local_denom_diff[local_denom_diff == 0] = 1
        # local_num_diff = np.add(channel_y, -local_min)
        # local_image_rgb[:, :, 0: 1] = np.divide(local_num_diff, local_denom_diff)
        return local_image_rgb
