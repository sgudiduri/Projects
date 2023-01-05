# The Keras model loading function does not play well with
# Pathlib at the moment, so we are using the old os module
# style
import os

PWD = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(PWD, '..'))
TRAINED_MODEL_DIR = os.path.join(PACKAGE_ROOT, 'trained_models')
TESTING_IMAGES = os.path.join(PACKAGE_ROOT, 'testing_images')

# MODEL FITTING
IMAGE_SIZE = 32 # 32 for testing, 150 for final model

#Hyperparameters
BATCH_SIZE = 128
OPTIMIZER = "adam"
EPOCHS = 10  # 1 for testing, 10 for final model

# MODEL PERSISTING
MODEL_NAME = 'cnn_model'
MODEL_PATH = "cnn_model.h5"
