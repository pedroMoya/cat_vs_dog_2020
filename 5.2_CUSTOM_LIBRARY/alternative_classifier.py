# the classifier derived from alternative training
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2


class alternative_classifier():

    def think(self, local_classifier_image_pil, local_aclass_settings, alt_classifier_):
        input_size = (260, 260)
        local_classifier_image = local_classifier_image_pil.resize(input_size, Image.ANTIALIAS)
        local_classifier_image = np.array(local_classifier_image, dtype=np.float32)
        local_classifier_image = local_classifier_image.reshape(1, local_classifier_image.shape[0],
                                                                local_classifier_image.shape[1],
                                                                local_classifier_image.shape[2])
        pred = alt_classifier_.predict(local_classifier_image)
        local_weights_hidden_message = np.load(''.join([local_aclass_settings['clean_data_path'],
                                                        'weights_hidden_message.npy']),
                                               allow_pickle=True)
        local_weights_hidden_message = local_weights_hidden_message[:][:][0]
        local_weights_no_hidden_message = np.load(''.join([local_aclass_settings['clean_data_path'],
                                                           'weights_no_hidden_message.npy']),
                                                  allow_pickle=True)
        local_weights_no_hidden_message = local_weights_no_hidden_message[:][:][0]
        pred_hidden = np.multiply(pred, local_weights_hidden_message).sum()
        pred_no_hidden = np.multiply(pred, local_weights_no_hidden_message).sum()
        pred = pred_hidden / (pred_hidden + pred_no_hidden)
        return pred
