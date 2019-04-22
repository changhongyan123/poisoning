
from matplotlib import pyplot as plt
from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations
from keras.models import load_model,Sequential, Model,model_from_json
from keras.datasets import mnist
from keras import backend as K
import keras
import numpy as np
from keras import backend as K

def load_model(filename):
    weight_file = 'model/weight/'+filename+'.h5'
    model_file = 'model/model_structure/'+filename+'.json'
    with open(model_file, 'r') as f:
        model = model_from_json(f.read())
    # Load weights into the new model
    model.load_weights(weight_file)
    return model

def load_plot_data():
    #load test dataset
    num_classes = 10

    # input image dimensions
    img_rows, img_cols = 28, 28
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train,y_train,x_test,y_test

def plot_image(x_train,y_train,x_test,y_test,model):
    class_idx = 0
    indices = np.where(y_test[:, class_idx] == 1.)[0]
    idx = indices[0]
    # Lets sanity check the picked image.


    layer_idx = utils.find_layer_idx(model, 'preds')

    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    grads = visualize_saliency(model, layer_idx, filter_indices=class_idx, seed_input=x_test[idx])
    plt.imshow(grads, cmap='jet')

    for modifier in ['guided', 'relu']:
        grads = visualize_saliency(model, layer_idx, filter_indices=class_idx,
                                   seed_input=x_test[idx], backprop_modifier=modifier)
        plt.figure()
        plt.title(modifier)
        plt.imshow(grads, cmap='jet')

        grads = visualize_saliency(model, layer_idx, filter_indices=class_idx, seed_input=x_test[idx],
                               backprop_modifier='guided', grad_modifier='negate')
        plt.imshow(grads, cmap='jet')
    plt.show()


    for class_idx in np.arange(10):
        indices = np.where(y_test[:, class_idx] == 1.)[0]
        idx = indices[0]

        f, ax = plt.subplots(1, 4)
        ax[0].imshow(x_test[idx][..., 0])

        for i, modifier in enumerate([None, 'guided', 'relu']):
            grads = visualize_saliency(model, layer_idx, filter_indices=class_idx,
                                       seed_input=x_test[idx], backprop_modifier=modifier)
            if modifier is None:
                modifier = 'vanilla'
            ax[i+1].set_title(modifier)
            ax[i+1].imshow(grads, cmap='jet')
    plt.show()


if __name__ == '__main__':
    x_train,y_train,x_test,y_test = load_plot_data()
    # model = load_model('attack_modelCNNmnistloss0.01')
    model = load_model('clean_modelCNNmnist')

    plot_image(x_train,y_train,x_test,y_test,model)
