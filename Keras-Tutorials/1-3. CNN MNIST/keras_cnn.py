import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
import os

def getData():
    """Loading in the MNIST dataset"""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    img_rows, img_cols = 28, 28

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    plt.imshow(X_train[0][:,:,0])
    plt.show()
    return X_train, y_train, X_test, y_test

def trainModel(X_train, y_train, X_test, y_test):
    """Creating and training a convolutional neural network"""
    batch_size = 64
    epochs = 15

    model = Sequential() # create sequential model
    
    # create some conv, maxpoll, dropout blocks
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(10, activation='softmax'))
    
    # specify a optimizer
    optimizer = RMSprop(lr=0.001) 
    
    # create a callback function
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
    
    # create a datagenerator for augmenting images
    datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1)
    
    # compiling and training the model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), epochs=epochs,
                                  validation_data=(X_test, y_test), verbose=2,
				                  steps_per_epoch=X_train.shape[0]//batch_size,
				                  callbacks=[learning_rate_reduction])
    # saving the model
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('mnist_model.h5')
    return model

def loadModel():
    """Load saved model"""
    json_file = open('model.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("mnist_model.h5")
    return model

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = getData()

    if(not os.path.exists('mnist_model.h5')):
        model = trainModel(X_train, y_train, X_test, y_test)
        print('trained model')
        print(model.summary())
    else:
        model = loadModel()
        print('loaded model')
        print(model.summary())
