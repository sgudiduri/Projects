import cv2
import tensorflow as tf
import config

    
def im_resize(im):
    im = cv2.resize(im, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    return im

def convert_to_tensor(im):
  return tf.expand_dims(tf.convert_to_tensor(im, dtype=tf.float32), axis=0)


