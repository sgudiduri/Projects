import pytest
import os

#from cifar_10_model.config import config


@pytest.fixture
def cat_dir():
    test_data_dir = os.path.join(config.DATASET_DIR, 'testing_images')
    cat_dir = os.path.join(test_data_dir, 'cat')

    return cat_dir


@pytest.fixture
def autombile_dir():
    test_data_dir = os.path.join(config.DATASET_DIR, 'testing_images')
    autombile_dir = os.path.join(test_data_dir, 'automobile')

    return autombile_dir
