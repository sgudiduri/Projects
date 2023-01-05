import cv2
import tensorflow as tf

#Loading config module.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

from config import config
    
def im_resize(im):
    im = cv2.resize(im, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    return im

def convert_to_tensor(im):
  return tf.expand_dims(tf.convert_to_tensor(im, dtype=tf.float32), axis=0)


