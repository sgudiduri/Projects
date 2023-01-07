import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

from cifar_10_model.config.config import Config
from cifar_10_model.predict import Predict
from cifar_10_model.processing.data_management import DataService
from cifar_10_model.processing.preprocessors import Preprocessor

import cv2


def test_make_prediction_on_cat(cat_dir):
    # Given or Arrange
    filename = 'cat.jpg'
    key = "cat"
    expected_classification = {'cat': 100.0}

    # When or Act
    c = Config()
    dm = DataService(c.IMAGE_SIZE, c.BATCH_SIZE,c.TRAINED_MODEL_DIR, c.MODEL_PATH)
    p = Preprocessor(c.IMAGE_SIZE)
    pred = Predict(dm,p)
    img = cv2.imread(f"{os.path.join(cat_dir,filename)}", cv2.IMREAD_COLOR) 
    results = pred.get_image_results(img)

    # Then or Assert
    assert results is not None
    assert key in results
    assert list(results.keys())[0] == list(expected_classification.keys())[0]
    assert results[key] == expected_classification[key]

def test_make_prediction_on_automobile(autombile_dir):
    # Given or Arrange
    filename = 'automobile.jpg'
    key = "automobile"
    expected_classification = {'automobile': 100.0}

    # When or Act
    c = Config()
    dm = DataService(c.IMAGE_SIZE, c.BATCH_SIZE,c.TRAINED_MODEL_DIR, c.MODEL_PATH)
    p = Preprocessor(c.IMAGE_SIZE)
    pred = Predict(dm,p)
    img = cv2.imread(f"{os.path.join(autombile_dir,filename)}", cv2.IMREAD_COLOR) 
    results = pred.get_image_results(img)

    # Then or Assert
    assert results is not None
    assert key in results
    assert list(results.keys())[0] == list(expected_classification.keys())[0]
    assert results[key] == expected_classification[key]

if __name__ == '__main__': 
    config = Config()
    test_data_dir = os.path.join(config.TESTING_IMAGES, 'cat')
    test_make_prediction_on_cat(test_data_dir)