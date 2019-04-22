
from matplotlib import pyplot as plt
from keras import activations
from keras.models import load_model,Sequential, Model,model_from_json
from keras.datasets import mnist
from keras import backend as K
import keras
import numpy as np
from keras import backend as K
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def load_model(filename):
    weight_file = 'model/weight/'+filename+'.h5'
    model_file = 'model/model_structure/'+filename+'.json'
    with open(model_file, 'r') as f:
        model = model_from_json(f.read())
    # Load weights into the new model
    model.load_weights(weight_file)
    return model

def get_dataset():
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


    return x_train,y_train,x_test,y_test


def get_feature(model,X):
    inp = model.input                    # input placeholder
    outputs = model.layers[-1].output    #last layer
    functors = K.function([inp, K.learning_phase()], [outputs])    # evaluation functions
    feature = functors([X, 1.])
    return feature[0]


def evaluation_linear_classifier(x_train,y_train,x_test,y_test):
    clf = LogisticRegression().fit(x_train,y_train)
    predicted = clf.predict(x_test)
    report = classification_report(y_test, predicted)
    print(report)
    return report


if __name__ == '__main__':

    x_train,y_train,x_test,y_test = get_dataset()
    x_train = x_train.reshape(x_train.shape[0],-1)
    x_test = x_test.reshape(x_test.shape[0],-1)
    attack_list = ['gradient_ascent','generative','lable_attack','random_attack']#,
    dataset_name = 'mnist'
    model_name = 'MLP'

    for training_method in ['normal','online','transfer']:
        for f in [0.1,0.2,0.3,0.4,0.5]:
            name = 'clean_' + training_method + dataset_name + model_name
            clean_model = load_model(name)
            x_train_feature = get_feature(clean_model,x_train)
            x_test_feature = get_feature(clean_model,x_test)
            print('-----------------------------------')
            print('clean model report: ')
            print('training method: ' + training_method)
            evaluation_linear_classifier(x_train_feature,y_train,x_test_feature,y_test)

            for attack in attack_list:
                name =  'attack_'+training_method + dataset_name+ model_name + attack + str(f)
                poisoning_model = load_model(name)
                x_train_feature = get_feature(poisoning_model,x_train)
                x_test_feature = get_feature(poisoning_model,x_test)
                print('-----------------------------------')
                print('evaluation of the attack model:')
                print('training method: ' + training_method)
                print('attack type: ' + attack)
                print('attack fraction: ' + str(f))
                evaluation_linear_classifier(x_train_feature,y_train,x_test_feature,y_test)
                print('-----------------------------------')
