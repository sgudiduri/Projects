import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))


from cifar_10_model import __version__ as _version
from cifar_10_model.predict import (get_image_results)

import cv2


def test_make_prediction_on_cat(cat_dir):
    # Given or Arrange
    filename = 'cat.jpg'
    key = "cat"
    expected_classification = {'cat': 100.0}

    # When or Act
    img = cv2.imread(f"{os.path.join(cat_dir,filename)}", cv2.IMREAD_COLOR) 
    results = get_image_results(img)

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
    img = cv2.imread(f"{os.path.join(autombile_dir,filename)}", cv2.IMREAD_COLOR) 
    results = get_image_results(img)

    # Then or Assert
    assert results is not None
    assert key in results
    assert list(results.keys())[0] == list(expected_classification.keys())[0]
    assert results[key] == expected_classification[key]