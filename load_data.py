from keras.datasets import mnist,fashion_mnist,cifar10,cifar100

def add_channel(X_train,X_test,data_format):

    if data_format == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    else:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], X_train.shape[2], 1)
    return X_train,X_test


def load_dataset(type,data_format='channels_last'):
    if type == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train,x_test = add_channel(x_train,x_test,data_format)
    elif type == 'fashion mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train,x_test = add_channel(x_train,x_test,data_format)
    elif type == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif type == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    dataset = {}
    dataset["clean_test"] = {}
    dataset["clean_train"] = {}

    dataset["clean_test"]["X"] = x_test
    dataset["clean_test"]["Y"] = y_test
    dataset["clean_train"]["X"] = x_train
    dataset["clean_train"]["Y"] = y_train

    return dataset
