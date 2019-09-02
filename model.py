from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Flatten, Dropout, Dense
from keras import optimizers


def get_model(img_x, img_y):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(img_x, img_y, 1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(128, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(256, kernel_size=(5, 5),  padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(8, activation='softmax'))

    adam = optimizers.sgd(lr=0.01, momentum=0.9)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  # loss='mse',
                  metrics=['accuracy'])

    return model


def get_mnist_model(img_x, img_y):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(img_x, img_y, 1)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    adam = optimizers.adam(lr=0.007, decay=1e-6)
    model.compile(optimizer=adam,
                  # loss='categorical_crossentropy',
                  loss='mse',
                  metrics=['accuracy'])
    return model