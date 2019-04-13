
from keras.utils import to_categorical
import numpy as np
from attack import posioning_attack
from utility import convert_dataset
import numpy as np
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


def generate_poisoning(model,dataset,training_info,parameters,functions):
    poisoning_data = posioning_attack(training_info['poisoning_type'],dataset,model,training_info['poisoning_fraction'],training_info,is_load = False)
    return poisoning_data


def compute_regret(model,X,Y):

    predicted_classes = model.predict_classes(X, verbose=0)

    correct_indices = np.nonzero(predicted_classes == Y)[0]
    incorrect_indices = np.nonzero(predicted_classes != Y)[0]

    return (correct_indices, incorrect_indices)



def training_ml_mdoel(dataset,training_info):
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

    classifiers = [
        linear_model.SGDClassifier(max_iter=100, tol=1e-3)
        ]

    x_train = dataset['clean_train']['X']
    y_train = dataset['clean_train']['Y']
    x_test = dataset['clean_test']['X']
    y_test = dataset['clean_test']['Y']
    x_train,y_train,x_test,y_test =convert_dataset('MLP',x_train,y_train,x_test,y_test)
    if training_info['is_poisoning'] == True:
        dataset = posioning_attack(training_info['poisoning_type'],dataset,None,training_info['poisoning_fraction'],training_info,is_load=True)
        x_p = dataset["poisoning_data"]['X']
        y_p = dataset["poisoning_data"]['Y']

        x_p = x_p.reshape(-1,x_train.shape[1])

        print(y_train.shape)
        y_p = y_p.reshape(-1,y_train.shape[1])

        x_train = np.concatenate((x_train, x_p), axis=0)
        y_train = np.concatenate((y_train, y_p), axis=0)
        y_train = np.argmax(y_train, axis=-1)

    for name, clf in zip(names, classifiers):
        print('start to train' + name)
        model = OneVsOneClassifier(clf).fit(x_train, y_train)
        # clf.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        print(name+': ' + score)


def training_network(model,dataset,training_info,parameters,functions):
    x_train = dataset['clean_train']['X']
    y_train = dataset['clean_train']['Y']
    x_test = dataset['clean_test']['X']
    y_test = dataset['clean_test']['Y']
    x_train,y_train,x_test,y_test =convert_dataset(training_info['model_name'],x_train,y_train,x_test,y_test)

    if training_info['is_poisoning'] == True:
        dataset = posioning_attack(training_info['poisoning_type'],dataset,model,training_info['poisoning_fraction'],training_info,is_load=True)
        x_p = dataset["poisoning_data"]['X']
        y_p = dataset["poisoning_data"]['Y']
        if training_info['model_name'] == 'MLP':
            x_p = x_p.reshape(-1,x_train.shape[1])
        else:
            x_p = x_p.reshape(-1,x_train.shape[1],x_train.shape[2],x_train.shape[3])

        print(y_train.shape[1])
        y_p = y_p.reshape(-1,y_train.shape[1])
        print(x_train.shape)
        print(x_p.shape)
        x_train = np.concatenate((x_train, x_p), axis=0)
        y_train = np.concatenate((y_train, y_p), axis=0)
        print('start to train')
        print(x_train.shape)

    model.fit(x_train,y_train, epochs=training_info['training_epoch'], batch_size=training_info['batch_size'],verbose = training_info['print_process'])
    training_score = model.evaluate(x_train, y_train, batch_size=training_info['batch_size'],verbose = training_info['print_process'])
    score = model.evaluate(x_test, y_test, batch_size=training_info['batch_size'],verbose = training_info['print_process'])


    print('--------------Training Information---------------')
    if training_info['print_model'] == True:
        print(model.summary())
    print('Input shape: '+ str(parameters['input_shape']))
    print('Output shape: '+ str(parameters['output_shape']))
    print('dataset: ' + training_info['dataset_name'])
    print('model name: ' + training_info['model_name'])
    print('learning_rate: '+ str(training_info['learning_rate']))
    print('batch_size: '+ str(training_info['batch_size']))
    print('learning epochs: '+ str(training_info['training_epoch']))
    print('training loss: %.3f' %training_score[0])
    print('test loss: %.3f' % score[0])
    print('training acc: %.3f' %training_score[1])
    print('test acc: %.3f' %score[1])
    print('optimizer: ' + functions['optimizer'] )
    print('Is poisoning attack: '+ str(training_info['is_poisoning']))
    if training_info['is_poisoning'] == True:
        print('poisoning attack:' + training_info['poisoning_type'])
        print('poisoning fraction: '+ str(training_info['poisoning_fraction']))
        if training_info['poisoning_type'] == 'gradient ascent':
            print('poisoning degree: '+ str(training_info['posioned_round']))
    print('-------------------------------------------------')

    return model

def online_training_network(model,dataset,training_info,parameters,functions):
    x_train = dataset['clean_train']['X']
    y_train = dataset['clean_train']['Y']
    x_test = dataset['clean_test']['X']
    y_test = dataset['clean_test']['Y']
    x_train,y_train,x_test,y_test =convert_dataset(training_info['model_name'],x_train,y_train,x_test,y_test)

    if training_info['is_poisoning'] == True:
        dataset = posioning_attack(training_info['poisoning_type'],dataset,model,training_info['poisoning_fraction'],training_info,is_load=True)
        x_p = dataset["poisoning_data"]['X']
        y_p = dataset["poisoning_data"]['Y']
        if training_info['model_name'] == 'MLP':
            x_p = x_p.reshape(-1,x_train.shape[1])
        else:
            x_p = x_p.reshape(-1,x_train.shape[1],x_train.shape[2],x_train.shape[3])

        # print(y_train.shape[1])
        y_p = y_p.reshape(-1,y_train.shape[1])
        # print(x_train.shape)
        # print(x_p.shape)
        #
        # print('start to train')
        # print(x_train.shape
    else:
        x_p = x_train[:600]
        y_p = y_train[:600]
    # x_train = np.concatenate((x_train, x_p), axis=0)
    # y_train = np.concatenate((y_train, y_p), axis=0)

    model.fit(x_train,y_train, epochs=training_info['training_epoch'], batch_size=training_info['batch_size'],verbose = training_info['print_process'])
    for i in range(x_p.shape[0]):
        model.fit(x_p[i:i+1],y_p[i:i+1], epochs=1, batch_size=1,verbose = training_info['print_process'])
        score = model.evaluate(x_test, y_test, batch_size=training_info['batch_size'],verbose = 0)
        # print('online learning score' + str(score))
    
    training_score = model.evaluate(x_train, y_train, batch_size=training_info['batch_size'],verbose = training_info['print_process'])
    score = model.evaluate(x_test, y_test, batch_size=training_info['batch_size'],verbose = training_info['print_process'])

    print('--------------Training Information---------------')
    if training_info['print_model'] == True:
        print(model.summary())
    print('Input shape: '+ str(parameters['input_shape']))
    print('Output shape: '+ str(parameters['output_shape']))
    print('dataset: ' + training_info['dataset_name'])
    print('model name: ' + training_info['model_name'])
    print('learning_rate: '+ str(training_info['learning_rate']))
    print('batch_size: '+ str(training_info['batch_size']))
    print('learning epochs: '+ str(training_info['training_epoch']))
    print('training loss: %.3f' %training_score[0])
    print('test loss: %.3f' % score[0])
    print('training acc: %.3f' %training_score[1])
    print('test acc: %.3f' %score[1])
    print('optimizer: ' + functions['optimizer'] )
    print('Is poisoning attack: '+ str(training_info['is_poisoning']))
    if training_info['is_poisoning'] == True:
        print('poisoning attack:' + training_info['poisoning_type'])
        print('poisoning fraction: '+ str(training_info['poisoning_fraction']))
        if training_info['poisoning_type'] == 'gradient ascent':
            print('poisoning degree: '+ str(training_info['posioned_round']))
    print('-------------------------------------------------')

    return model
