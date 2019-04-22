
from keras.utils import to_categorical
import numpy as np
import random
from attack import posioning_attack
from utility import *
import numpy as np
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.multiclass import OneVsOneClassifier
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

def generate_poisoning(model,dataset,training_info,parameters,functions):
    # generate the poisong data without training.

    poisoning_data = posioning_attack(training_info['poisoning_type'],dataset,model,training_info['poisoning_fraction'],training_info,is_load = False)
    return poisoning_data


def call_training(model,dataset,training_info,parameters,functions):
    # call the training method based on the parameter

    if training_info['training_type'] == 'normal':
        model,history = training_network(model,dataset,training_info,parameters,functions)
    elif training_info['training_type'] == 'online':
        model,history = online_training_network(model,dataset,training_info,parameters,functions)
    elif training_info['training_type'] == 'transfer':
        model,history = transfer_learning(model,dataset,training_info,parameters,functions)
    elif training_info['training_type'] == 'classification':
        model,history = training_ml_mdoel(model,dataset,training_info,parameters,functions)

    return model,history



def training_ml_mdoel(model,dataset,training_info,parameters,functions):
    data = load_and_generate_dataset(model,dataset,training_info,parameters,functions)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    print(x_train.shape)
    print(y_train.shape)
    print('ML Model')





def training_network(model,dataset,training_info,parameters,functions):
    # normal learning: trained the whole model with the poisoned dataset
    # input: model, dataset, training information
    # output: trained model

    data = load_and_generate_dataset(model,dataset,training_info,parameters,functions)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    x_train = np.concatenate((x_train, data['x_p']), axis=0)
    y_train = np.concatenate((y_train, data['y_p']), axis=0)
    overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience = 20)

    history = model.fit(x_train,y_train, validation_data = (x_test,y_test),epochs=10000000, callbacks=[overfitCallback], batch_size=training_info['batch_size'],verbose = training_info['print_process'])

    if training_info['plot_loss']:
         filename = training_info['dataset_name']+'_'+training_info['model_name']
         if training_info['is_poisoning']:
             filename = filename + '_' + training_info['poisoning_type'] + '_' + str(training_info['poisoning_fraction'])
         save_history(history,filename)

    train_acc = history.history['acc'][-1]
    acc = history.history['val_acc'][-1]
    train_loss = history.history['loss'][-1]
    loss = history.history['val_loss'][-1]

    print_information(model,training_info,parameters,functions,train_acc,acc,train_loss,loss)
    return model,history.history

def online_training_network(model,dataset,training_info,parameters,functions):
    # online learning: train the whole model on the clean dataset and update the model by each example based on the online sgd.
    # input: model, dataset, training information
    # output: trained model
    data = load_and_generate_dataset(model,dataset,training_info,parameters,functions)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    x_p = data['x_p']
    y_p = data['y_p']

    x_train = np.concatenate((x_train, data['x_p']), axis=0)
    y_train = np.concatenate((y_train, data['y_p']), axis=0)


    x_train, y_train = shuffle(x_train, y_train, random_state=0)

    overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience = 20)

    history = model.fit(x_train,y_train, validation_data = (x_test,y_test),epochs=100, callbacks=[overfitCallback], batch_size=1,verbose = training_info['print_process'])


    train_acc = history.history['acc'][-1]
    acc = history.history['val_acc'][-1]
    train_loss = history.history['loss'][-1]
    loss = history.history['val_loss'][-1]


    print_information(model,training_info,parameters,functions,train_acc,acc,train_loss,loss)
    return model,history.history


def transfer_learning(model,dataset,training_info,parameters,functions):
    # transfer learning: train the whole model on the clean dataset and fix the all the layers except the last layer.
    # input: model, dataset, training information
    # output: trained model
    data = load_and_generate_dataset(model,dataset,training_info,parameters,functions)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience = 20)

    history = model.fit(x_train,y_train, validation_data = (x_test,y_test),epochs=10000000, callbacks=[overfitCallback], batch_size=training_info['batch_size'],verbose = training_info['print_process'])


    print('before training loss: %.3f' %history.history['loss'][-1])
    print('before test loss: %.3f' % history.history['val_loss'][-1])
    print('before training acc: %.3f' %history.history['acc'][-1])
    print('before test acc: %.3f' %history.history['acc'][-1])

    for layer in model.layers[:-1]:
        layer.trainable = False
    model.compile(optimizer=functions['optimizer'],loss=functions['loss'],metrics= functions['metrics'])

    x_train = np.concatenate((x_train, data['x_p']), axis=0)
    y_train = np.concatenate((y_train, data['y_p']), axis=0)

    overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience = 20)

    history = model.fit(x_train,y_train, validation_data = (x_test,y_test),epochs=10000000, callbacks=[overfitCallback], batch_size=training_info['batch_size'],verbose = training_info['print_process'])

    train_acc = history.history['acc'][-1]
    acc = history.history['val_acc'][-1]
    train_loss = history.history['loss'][-1]
    loss = history.history['val_loss'][-1]

    print_information(model,training_info,parameters,functions,train_acc,acc,train_loss,loss)
    return model,history.history



def load_and_generate_dataset(model,dataset,training_info,parameters,functions):
    # combine the clean dataset and poisoning dataset
    # input: model,clean dataset, training information
    # output: combined dataset
    x_train = dataset['clean_train']['X']
    y_train = dataset['clean_train']['Y']
    x_test = dataset['clean_test']['X']
    y_test = dataset['clean_test']['Y']
    x_train,y_train,x_test,y_test =convert_dataset(training_info['model_name'],x_train,y_train,x_test,y_test,training_info)

    if training_info['is_poisoning'] == True:
        dataset = posioning_attack(training_info['poisoning_type'],dataset,model,training_info['poisoning_fraction'],training_info,is_load=True)
        x_p = dataset["poisoning_data"]['X']
        y_p = dataset["poisoning_data"]['Y']

        if training_info['model_name'] == 'MLP':
            x_p = x_p.reshape(-1,x_train.shape[1])
        else:
            x_p = x_p.reshape(-1,x_train.shape[1],x_train.shape[2],x_train.shape[3])
    else:
        poisoning_samples = int(training_info['poisoning_fraction'] * x_train.shape[0])
        x_p = x_train[:poisoning_samples]
        y_p = y_train[:poisoning_samples]


    random_index = np.random.randint(0, x_train.shape[0],x_p.shape[0])

    x_train = np.delete(x_train, random_index,axis=0)
    y_train = np.delete(y_train, random_index,axis=0)

    data = {}
    data['x_train'] = x_train
    data['y_train'] = y_train
    data['x_test'] = x_test
    data['y_test'] = y_test
    data['x_p'] = x_p
    data['y_p'] = y_p
    return data

def print_information(model,training_info,parameters,functions,train_acc,acc,train_loss,loss):
    #print the training information

    print('--------------Training Information---------------')
    if training_info['print_model'] == True:
        print(model.summary())
    print('Input shape: '+ str(parameters['input_shape']))
    print('Output shape: '+ str(parameters['output_shape']))
    print('dataset: ' + training_info['dataset_name'])
    print('model name: ' + training_info['model_name'])
    print('learning_rate: '+ str(training_info['learning_rate']))
    print('batch_size: '+ str(training_info['batch_size']))
    # print('learning epochs: '+ str(training_info['training_epoch']))

    print('training loss: %.3f' %train_loss)
    print('test loss: %.3f' % loss)
    print('training acc: %.3f' %train_acc)
    print('test acc: %.3f' %acc)
    print('optimizer: ' + functions['optimizer'] )
    print('training method: ' + training_info['training_type'])
    print('Is poisoning attack: '+ str(training_info['is_poisoning']))
    if training_info['is_poisoning']:
        print('poisoning attack:' + training_info['poisoning_type'])
        print('poisoning fraction: '+ str(training_info['poisoning_fraction']))
        if training_info['poisoning_type'] == 'gradient_ascent':
            print('poisoning degree: '+ str(training_info['posioned_round']))
    print('-------------------------------------------------')
