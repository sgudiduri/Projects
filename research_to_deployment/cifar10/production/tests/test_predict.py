from cifar_10_model import __version__ as _version
from cifar_10_model.predict import (get_image_results)

import cv2


def test_make_prediction_on_cat(cat_dir):
    # Given or Arrange
    filename = 'cat.png'
    expected_classification = {'cat': 100.0}

    # When or Act
    img = cv2.imread(f"{cat_dir}/{filename}", cv2.IMREAD_COLOR) 
    results = get_image_results(img)

    # Then or Assert
    assert results['predictions'] is not None
    assert results.keys()[0] == expected_classification.keys()[0]
    assert results['cat'] == expected_classification['cat']
    assert results['version'] == _version

def test_make_prediction_on_automobile(autombile_dir):
    # Given or Arrange
    filename = 'automobile.png'
    expected_classification = {'automobile': 100.0}

    # When or Act
    img = cv2.imread(f"{autombile_dir}/{filename}", cv2.IMREAD_COLOR) 
    results = get_image_results(img)

    # Then or Assert
    assert results['predictions'] is not None
    assert results.keys()[0] == expected_classification.keys()[0]
    assert results['automobile'] == expected_classification['automobile']
    assert results['version'] == _version