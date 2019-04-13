import keras
import numpy as np
from keras.utils import to_categorical


class LossHistory(keras.callbacks.Callback):
    def on_test_begin(self, logs={}):
        self.losses = []

    def on_test_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def convert_dataset(type,x_train,y_train,x_test,y_test):
    input_dim = x_test[0].shape
    output_dim = len(np.unique(y_train))

    y_test = to_categorical(y_test, num_classes=output_dim)
    y_train = to_categorical(y_train, num_classes=output_dim)

    #for the image input
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if type == 'MLP':
        input_dim = x_test[0].size

        x_train = x_train.reshape(-1,input_dim)
        output_dim = len(np.unique(y_train))
        x_test = x_test.reshape(-1,input_dim)

    return x_train,y_train,x_test,y_test
