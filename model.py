from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.vgg16 import VGG16


def MLP(hidden_layer,input_dim,out_dim,activate_fun,loss_fun,optimizer_fun,metrics):
    # parameters['hidden_layer'] = [20,10,2]
    # parameters['input_shape'] =(3,20,20)
    # parameters['output_shape'] = 1
    # functions['optimizer'] = 'rmsprop'
    # functions['loss'] = 'binary_crossentropy'
    # functions['activation'] = 'sigmoid'
    # metrics = ['accuracy']
    # print(create_model('MLP',parameters,functions,metrics).to_json())
    model = Sequential()
    for i in range(len(hidden_layer)):
        if i ==0:
            model.add(Dense(hidden_layer[i], activation='relu', input_dim=input_dim))
        else:
            model.add(Dense(hidden_layer[i], activation='relu'))
    model.add(Dense(out_dim, activation=activate_fun))
    model.compile(optimizer=optimizer_fun,loss=loss_fun,metrics=metrics)
    return model

def CNN(cnn_layer,fc_layer,input_shape,output_shape,activate_fun,loss_fun,optimizer_fun,metrics,data_format):
    # parameters = {}
    # functions ={}
    #
    # functions['optimizer'] = 'rmsprop'
    # functions['loss'] = 'binary_crossentropy'
    # functions['activation'] = 'sigmoid'
    # metrics = ['accuracy']
    # parameters['output_shape'] = 1
    # parameters['input_shape'] = (3,150,150)
    #
    #
    #
    # parameters['fc_layer'] = [20,10,2]
    # parameters['cnn_layer'] = {}
    # parameters['cnn_layer']['filter_number'] = [32,32,64]
    # parameters['cnn_layer']['filter_shape'] =[(3,3),(3,3),(3,3)]
    # parameters['cnn_layer']['pooling_shape'] = [(2,2),(2,2),(2,2)]
    # parameters['data_format'] = 'channels_first'

    model = Sequential()
    for i in range(len(cnn_layer)):
        if i == 0:
            model.add(Conv2D(cnn_layer['filter_number'][i], cnn_layer['filter_shape'][i], data_format=data_format,input_shape=input_shape))
        else:
            model.add(Conv2D(cnn_layer['filter_number'][i], cnn_layer['filter_shape'][i],data_format=data_format))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=cnn_layer['pooling_shape'][i],data_format=data_format))
    model.add(Flatten())

    for j in range(len(fc_layer)):
        model.add(Dense(fc_layer[j], activation='relu'))
    model.add(Dense(output_shape, activation=activate_fun))
    model.compile(optimizer=optimizer_fun,loss=loss_fun,metrics=metrics)
    return model

def VGG():
    model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    return model

def create_model(type,parameters,functions,metrics):

    input_dim = parameters['input_shape']
    out_dim = parameters['output_shape']
    activation = functions['activation']
    loss = functions['loss']
    optimizer = functions['optimizer']
    if type == 'MLP':
        hidden_layer = parameters['hidden_layer']
        model = MLP(hidden_layer,input_dim,out_dim,activation,loss,optimizer,metrics)
    elif type == 'CNN':
        fc_layer = parameters['fc_layer']
        cnn_layer = parameters['cnn_layer']
        data_format = parameters['data_format']
        model = CNN(cnn_layer,fc_layer,input_dim,out_dim,activation,loss,optimizer,metrics,data_format)
    elif type == 'VGG':
        model = VGG()
    #TODO: add more information about the networks

    return model




parameters = {}
functions ={}

functions['optimizer'] = 'rmsprop'
functions['loss'] = 'binary_crossentropy'
functions['activation'] = 'sigmoid'
metrics = ['accuracy']
parameters['output_shape'] = 1
parameters['input_shape'] = (3,150,150)



# print(create_model('VGG',parameters,functions,metrics).to_json())
