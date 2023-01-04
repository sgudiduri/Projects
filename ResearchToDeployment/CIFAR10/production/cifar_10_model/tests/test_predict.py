from cifar_10_model import __version__ as _version
from cifar_10_model.predict import (get_image_results)


def test_make_prediction_on_sample(cat_dir):
    # Given
    filename = '1.png'
    expected_classification = 'Charlock'

    # When
    results = get_image_results(image_directory=cat_dir,
                                     image_name=filename)

    # Then
    assert results['predictions'] is not None
    assert results['readable_predictions'][0] == expected_classification
    assert results['version'] == _version
