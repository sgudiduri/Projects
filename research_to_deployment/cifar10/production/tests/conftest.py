import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

from cifar_10_model.config import config
import pytest


@pytest.fixture
def cat_dir():
    test_data_dir = os.path.join(config.TESTING_IMAGES, 'cat')
    return test_data_dir


@pytest.fixture
def autombile_dir():
    test_data_dir = os.path.join(config.TESTING_IMAGES, 'automobile')
    return test_data_dir


