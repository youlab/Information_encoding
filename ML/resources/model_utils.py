import numpy as np
from keras.initializers import glorot_normal
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Reshape
from keras.models import Sequential
from tqdm import tqdm


def build_model(nb_classes: int, image_length, seed=25):
    initializer = glorot_normal(seed=seed)
    model = Sequential()
    model.add(Reshape((image_length, image_length, 1), input_shape=(image_length, image_length,)))

    model.add(
        Conv2D(64, kernel_size=(5, 5), padding='same', kernel_initializer=initializer)）
    model.add(MaxPooling2D(pool_size=(10, 10)))  
    model.add(Activation('relu'))

    model.add(
        Conv2D(128, kernel_size=(5, 5), padding='same', kernel_initializer=initializer)）
    model.add(MaxPooling2D(pool_size=(8, 8)))  
    model.add(Activation('relu'))

    model.add(Reshape((128 * 1 * 1,), input_shape=(1, 1, 128)))
    model.add(Dense(50, activation='relu', kernel_initializer=initializer))
    model.add(Dense(nb_classes, activation='softmax', kernel_initializer=initializer))

    return model

def train_model_noitr(model, train_generator, valid_generator, epochs, early_stopping, verbose=0):
    model_history = model.fit_generator(generator=train_generator,
                                        epochs=epochs,
                                        verbose=verbose,
                                        validation_data=valid_generator,
                                        callbacks=[early_stopping])
     
    #if 'early_stopping.stopped_epoch' not in locals():
    stopped_epoch = [early_stopping.stopped_epoch]
    # else:
    #    stopped_epoch = 0
    
    val_acc_history  = model_history.history['val_accuracy']
    acc_history      = model_history.history['accuracy']
    val_loss_history = model_history.history['val_loss']
    loss_history     = model_history.history['loss']

    return model, model_history, val_loss_history, loss_history, val_acc_history, acc_history,stopped_epoch

def evaluate_model(model, evaluation_generator, workers):
    predictions = model.predict_generator(generator=evaluation_generator,
                                          workers=workers,
                                          use_multiprocessing=True)
    return predictions
