
from keras.utils import to_categorical
import numpy as np
from attack import posioning_attack



def convert_dataset(type,x_train,y_train,x_test,y_test):
    input_dim = x_test[0].shape
    output_dim = len(np.unique(y_train))

    y_test = to_categorical(y_test, num_classes=output_dim)
    y_train = to_categorical(y_train, num_classes=output_dim)

    if type == 'MLP':
        input_dim = input_dim[1]*input_dim[2]* input_dim[3]
        output_dim = len(np.unique(y_train))
        x_train = x_train.reshape(-1,input_dim)
        x_test = x_test.reshape(-1,input_dim)

    return x_train,y_train,x_test,y_test



def training_network(model,dataset,training_info,parameters,functions):

    if training_info['is_poisoning'] == True:
        dataset = posioning_attack(training_info['poisoning_type'],dataset,model,training_info['poisoning_fraction'])
        x_train = dataset['poisoning_train']['X']
        y_train = dataset['poisoning_train']['Y']
        x_test = dataset['clean_test']['X']
        y_test = dataset['clean_test']['Y']
        x_train,y_train,x_test,y_test =convert_dataset(training_info['model_name'],x_train,y_train,x_test,y_test)
    else:
        x_train = dataset['clean_train']['X']
        x_test = dataset['clean_test']['X']
        y_train = dataset['clean_train']['Y']
        y_test = dataset['clean_test']['Y']
        x_train,y_train,x_test,y_test = convert_dataset(training_info['model_name'],x_train,y_train,x_test,y_test)

    model.fit(x_train,y_train, epochs=training_info['training_epoch'], batch_size=training_info['batch_size'])

    score = model.evaluate(x_test, y_test, batch_size=training_info['batch_size'])


    print('--------------Training Information---------------')
    # print(model.summary())
    print('Input shape: '+ str(parameters['input_shape']))
    print('Output shape: '+ str(parameters['output_shape']))
    print('dataset: ' + training_info['dataset_name'])
    print('model name: ' + training_info['model_name'])
    print('learning_rate: '+ str(training_info['learning_rate']))
    print('batch_size: '+ str(training_info['batch_size']))
    print('learning epochs: '+ str(training_info['training_epoch']))
    print('evaluation :' + str(score))
    print('optimizer: ' + functions['optimizer'] )
    print('Is poisoning attack: '+ str(training_info['is_poisoning']))
    if training_info['is_poisoning'] == True:
        print('poisoning attack:' + training_info['poisoning_type'])
        print('poisoning fraction: '+ str(training_info['poisoning_fraction']))
    print('-------------------------------------------------')
