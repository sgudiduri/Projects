import processing.data_management as dm
import model as m
import config


def run_training(save_result: bool = True):

    (new_x_train, new_y_train), _, (x_val, y_val) = dm.load_data()

    train_dataset = dm.make_datasets(new_x_train, new_y_train.reshape(-1), is_train=True)
    val_dataset = dm.make_datasets(x_val, y_val.reshape(-1))       

    model = m.cnn_model()
    model.summary()

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        callbacks=[m.callbacks_list()],
    )
    
    if save_result:
        dm.save_pipeline_keras(model)

if __name__ == '__main__':
    run_training(save_result=True)