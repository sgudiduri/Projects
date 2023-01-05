from sklearn.model_selection import train_test_split

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import load_model

#Loading config module.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

from config import config

def load_data():    
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    new_x_train,  x_val, new_y_train, y_val = train_test_split(x_train, y_train, test_size=.2)
    return (new_x_train, new_y_train), (x_test, y_test), (x_val, y_val)

def make_datasets(images, labels, is_train=False):

    data_augmentation = keras.Sequential(
        [layers.RandomCrop(config.IMAGE_SIZE, config.IMAGE_SIZE), 
        layers.RandomFlip("horizontal"),],
        name="data_augmentation",
    )

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_train:
        dataset = dataset.shuffle(config.BATCH_SIZE * 10)
    dataset = dataset.batch(config.BATCH_SIZE)
    if is_train:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
    return dataset.prefetch(tf.data.AUTOTUNE)

def save_pipeline_keras(model):   
    path =  os.path.join(config.TRAINED_MODEL_DIR, config.MODEL_PATH)
    model.save(path)

def load_pipeline_keras():
   path =  os.path.join(config.TRAINED_MODEL_DIR, config.MODEL_PATH)
   return load_model(path)

if __name__ == '__main__':
            
    (new_x_train, new_y_train), (x_test, y_test), (x_val, y_val) = load_data()

    print(f"Training data samples: {len(new_x_train)}")
    print(f"Validation data samples: {len(x_val)}")
    print(f"Test data samples: {len(x_test)}")

    train_dataset = make_datasets(new_x_train, new_y_train.reshape(-1), is_train=True)
    val_dataset = make_datasets(x_val, y_val.reshape(-1))
    test_dataset = make_datasets(x_test, y_test.reshape(-1))
   
