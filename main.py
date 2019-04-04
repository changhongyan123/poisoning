from attack import posioning_attack
from model import create_model
from load_data import load_dataset
from train import convert_dataset,training_network
from attack import posioning_attack
import numpy as np
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset_list = ['mnist','fashion mnist','cifar10','cifar100']
attack_list = ['loss','lable attack','gradient ascent','min max','influence']


if __name__ == '__main__':



    training_info = {}
    training_info['poisoning_fraction'] = 0.2
    training_info['poisoning_type'] = 'lable attack'
    training_info['is_poisoning'] = True
    training_info['training_epoch'] = 10
    training_info['learning_rate'] = 0.01
    training_info['dataset_name'] = 'mnist'
    training_info['model_name'] = 'CNN'
    training_info['batch_size'] = 128

    dataset = load_dataset(training_info['dataset_name'])


    parameters = {}
    parameters['input_shape'] = dataset['clean_train']['X'][0].shape
    parameters['output_shape'] = len(np.unique(dataset['clean_train']['Y']))
    parameters['fc_layer'] = [20,10]
    parameters['cnn_layer'] = {}
    parameters['cnn_layer']['filter_number'] = [16,8,8]
    parameters['cnn_layer']['filter_shape'] =[(3,3),(3,3),(3,3)]
    parameters['cnn_layer']['pooling_shape'] = [(2,2),(2,2),(2,2)]
    parameters['data_format'] = 'channels_last'


    functions = {}
    functions['optimizer'] = 'adam'
    functions['loss'] = 'categorical_crossentropy'
    functions['activation'] = 'softmax'

    metrics = ['accuracy']

    #clean model
    training_info['is_poisoning'] = True
    model = create_model(training_info['model_name'],parameters,functions,metrics)
    training_network(model,dataset,training_info,parameters,functions)


    #poisoning model
    for f in [0.1,0.2,0.3,0.4,0.5]:
        model_p = create_model(training_info['model_name'],parameters,functions,metrics)
        training_info['is_poisoning'] = True
        training_network(model_p,dataset,training_info,parameters,functions)




    # parameters['hidden_layer'] = [20,10,10] MLP parameters
