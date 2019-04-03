
from keras.utils import to_categorical
import numpy as np


def convert_dataset(type,dataset):
    input_dim = dataset["clean_train"]["X"][0].shape
    output_dim = len(np.unique(dataset["clean_train"]["Y"]))

    x_train = dataset["clean_train"]["X"]
    x_test = dataset["clean_test"]["X"]
    y_train = dataset["clean_train"]["Y"]
    y_test = dataset["clean_test"]["Y"]
    y_test = to_categorical(y_test, num_classes=output_dim)
    y_train = to_categorical(y_train, num_classes=output_dim)

    if type == 'MLP':
        input_dim = input_dim[1]*input_dim[2]* input_dim[3]
        output_dim = len(np.unique(dataset["clean_train"]["Y"]))
        x_train = dataset["clean_train"]["X"].reshape(-1,input_dim)
        x_test = dataset["clean_test"]["X"].reshape(-1,input_dim)

    return x_train,y_train,x_test,y_test,input_dim,output_dim
