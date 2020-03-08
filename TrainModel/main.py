import os
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def texture_dataset(split_Train_Test=0.9):
    xAll = np.load('dataAll_0to1.npy')
    yAll = np.load('labelAll_0to1.npy')

    x = xAll[0:np.int(xAll.shape[0] * split_Train_Test), :, :]
    y = yAll[0:np.int(xAll.shape[0] * split_Train_Test)]

    x_val = xAll[np.int(xAll.shape[0] * split_Train_Test):-1, :, :]
    y_val = yAll[np.int(xAll.shape[0] * split_Train_Test):-1]
    return x, y, x_val, y_val


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    normed_train_data, train_labels, normed_test_data, test_labels = texture_dataset()
    if tf.keras.backend.image_data_format() == 'channel_first':
        normed_train_data = normed_train_data.reshape(
            normed_train_data.shape[0], 1, normed_train_data.shape[1], normed_train_data.shape[2])
        normed_test_data = normed_test_data.reshape(
            normed_test_data.shape[0], 1, normed_test_data.shape[1], normed_test_data.shape[2])
        input_shape = (
            1, normed_train_data.shape[1], normed_train_data.shape[2])
    else:
        normed_train_data = normed_train_data.reshape(
            normed_train_data.shape[0], normed_train_data.shape[1], normed_train_data.shape[2], 1)
        normed_test_data = normed_test_data.reshape(
            normed_test_data.shape[0], normed_test_data.shape[1], normed_test_data.shape[2], 1)
        input_shape = (normed_test_data.shape[1],
        normed_test_data.shape[2], 1)

    model = tf.keras.Sequential([layers.Reshape(target_shape=[normed_train_data.shape[1], normed_train_data.shape[2], 1], input_shape=input_shape),
    layers.Conv2D(30, (5, 5), strides=(3, 3),
                  padding='same', activation=tf.nn.elu),
    layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
    layers.Conv2D(60, (3, 3), strides=(2, 2),
                  padding='same', activation=tf.nn.elu),
    layers.Conv2D(60, (3, 3), padding='same', activation=tf.nn.elu), layers.MaxPooling2D((2, 2), (2, 2), padding='same'), layers.Conv2D(
        120, (3, 3), padding='same', activation=tf.nn.elu), layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
    layers.Flatten(),
    layers.Dense(110, activation=tf.nn.elu),
    layers.Dropout(rate=0.4),
    layers.Dense(1)])

    model.summary()

    model.compile(optimizer=optimizers.Adam(), loss='mse')
    normed_train_data, train_labels = shuffle(normed_train_data,
                                            train_labels)
    normed_test_data, test_labels = shuffle(normed_test_data,
                                          test_labels)
    history = model.fit(normed_train_data, train_labels, epochs=500, validation_data=(normed_test_data, test_labels))
    model.save('SaveModel.h5')

    return history


if __name__ == '__main__':
    model=main()
