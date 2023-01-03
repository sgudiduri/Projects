# for the convolutional network
import tensorflow as tf
import tensorflow_addons as tfa
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten,BatchNormalization
from keras.callbacks import ModelCheckpoint

import config


def cnn_model(kernel_size = (3,3),
              pool_size= (2,2),
              first_filters = 64,
              second_filters = 128,
              third_filters = 256,
              dropout_dense = 0.2):
        
    model = Sequential()

    #Layer1
    model.add(Conv2D(first_filters, kernel_size,strides=1, activation = 'relu', 
                    input_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)))
    model.add(BatchNormalization(momentum=0.15, axis=-1))
    model.add(Conv2D(first_filters, kernel_size, activation = 'relu', 
                    input_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)))
    model.add(BatchNormalization(momentum=0.15, axis=-1))
    model.add(MaxPooling2D(pool_size = pool_size, strides=2)) 

    #Layer2
    model.add(Conv2D(second_filters, kernel_size,strides=1, activation = 'relu', 
                    input_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)))
    model.add(BatchNormalization(momentum=0.15, axis=-1))
    model.add(Conv2D(second_filters, kernel_size, activation = 'relu', 
                    input_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)))
    model.add(BatchNormalization(momentum=0.15, axis=-1))
    model.add(MaxPooling2D(pool_size = pool_size, strides=2)) 

    #Layer3
    model.add(Conv2D(third_filters, kernel_size,strides=1, activation = 'relu', 
                    input_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE, 3))) 
    model.add(BatchNormalization(momentum=0.15, axis=-1))
    model.add(Conv2D(third_filters, kernel_size, activation = 'relu', 
                    input_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)))
    model.add(BatchNormalization(momentum=0.15, axis=-1))
    model.add(MaxPooling2D(pool_size=(1,1), strides=1))

    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(dropout_dense))
    model.add(Dense(10, activation = "softmax"))

    optimizer = tfa.optimizers.AdamW(
        learning_rate=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

def callbacks_list(moitor_mod = "val_accuracy", save_best=True, save_weights=True):

    return [ModelCheckpoint(
        config.MODEL_PATH,
        monitor=moitor_mod,
        save_best_only=save_best,
        save_weights_only=save_weights,
    )]         

                              
if __name__ == '__main__':
    
    model = cnn_model()
    model.summary()
    
    # import data_management as dm
    # (new_x_train, new_y_train), (x_test, y_test), (x_val, y_val) = dm.load_data()

    # print(f"Training data samples: {len(new_x_train)}")
    # print(f"Validation data samples: {len(x_val)}")
    # print(f"Test data samples: {len(x_test)}")

    # train_dataset = dm.make_datasets(new_x_train, new_y_train.reshape(-1), is_train=True)
    # val_dataset = dm.make_datasets(x_val, y_val.reshape(-1))
    # test_dataset = dm.make_datasets(x_test, y_test.reshape(-1))

    # model = cnn_model()
    # model.summary()
    # history = model.fit(
    #     train_dataset,
    #     validation_data=val_dataset,
    #     epochs=config.EPOCHS,
    #     callbacks=[callbacks_list()],
    # )
    
