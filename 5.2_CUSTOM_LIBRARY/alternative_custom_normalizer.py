# function that makes last pre-process step to images, preparing data for training the CNNs
import numpy as np
import tensorflow as tf
import cv2


class alternative_custom_image_normalizer():

    def normalize(self, local_image_rgb):
        local_max = np.amax(local_image_rgb, axis=(0, 1))
        local_min = np.amin(local_image_rgb, axis=(0, 1))
        local_denom_diff = np.add(local_max, -local_min)
        local_denom_diff[local_denom_diff == 0] = 1
        local_num_diff = np.add(local_image_rgb, -local_min)
        return np.divide(local_num_diff, local_denom_diff)
