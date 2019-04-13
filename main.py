import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from attack import posioning_attack
from model import create_model
from load_data import load_dataset
from train import *
from attack import posioning_attack
import numpy as np
import keras
import os
import tensorflow as tf
from utility import LossHistory

import keras

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2"


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset_list = ['mnist','fashion mnist','cifar10','cifar100']
attack_list = ['loss','lable attack','gradient ascent','generative','random attack']


if __name__ == '__main__':



    training_info = {}
    training_info['poisoning_fraction'] = 0.2
    training_info['poisoning_type'] = 'lable attack'
    training_info['is_poisoning'] = True
    training_info['training_epoch'] = 10
    training_info['learning_rate'] = 0.01
    training_info['dataset_name'] = 'mnist'
    training_info['model_name'] = 'CNN' #'CNN'
    training_info['batch_size'] = 256
    training_info['print_model'] = False
    training_info['print_process'] = 0


    dataset = load_dataset(training_info['dataset_name'])


    parameters = {}
    if training_info['model_name'] == 'MLP':
        parameters['input_shape'] = dataset['clean_train']['X'][0].size
    else:
        parameters['input_shape'] = dataset['clean_train']['X'][0].shape
    parameters['output_shape'] = len(np.unique(dataset['clean_train']['Y']))
    # parameters['fc_layer']['kernel_init'] = 'random_uniform'
    # parameters['fc_layer']['bias_init'] = 'ones'
    parameters['hidden_layer'] = [64,10]# MLP parameters
    parameters['fc_layer'] = [256,128,10]
    parameters['cnn_layer'] = {}
    # parameters['cnn_layer']['kernel_init'] = 'glorot_uniform'
    # parameters['cnn_layer']['bias_init'] = 'zeros'
    parameters['cnn_layer']['padding'] = 'SAME'
    parameters['cnn_layer']['filter_number'] = [18,18]
    parameters['cnn_layer']['filter_shape'] =[(14,14),(7,7)]
    parameters['cnn_layer']['pooling_shape'] = [(2,2),(2,2)]
    parameters['data_format'] = 'channels_last'


    functions = {}
    functions['optimizer'] = 'sgd'
    functions['loss'] = 'categorical_crossentropy'
    functions['activation'] = 'softmax'
    metrics = ['accuracy']


    # training a new  model
    # training_info['is_poisoning'] = False
    # model = create_model(training_info['model_name'],parameters,functions,metrics)
    # training_network(model,dataset,training_info,parameters,functions)

    # for f in [0.1,0.2,0.3]:
    # training_info['is_poisoning'] = True
    # training_ml_mdoel(dataset,training_info)

    # training on different attack
    for attack in attack_list[1:]:
        training_info['poisoning_fraction'] = 0.1
        training_info['poisoning_type'] = attack
        if attack == 'gradient ascent':
            training_info['posioned_round'] = 4
        elif attack == 'generative':
            training_info['posioned_round'] = 9
        model_p = create_model(training_info['model_name'],parameters,functions,metrics)
        # generate_poisoning(model_p,dataset,training_info,parameters,functions)
        online_training_network(model_p,dataset,training_info,parameters,functions)


    # for f in [0.1,0.2,0.3]:
    #     for type in ['loss','lable attack']:
    #         training_info['poisoning_fraction'] = f
    #         training_info['poisoning_type'] = type
    #         model_p = create_model(training_info['model_name'],parameters,functions,metrics)
    #         # generate_poisoning(model_p,dataset,training_info,parameters,functions)
    #         training_network(model_p,dataset,training_info,parameters,functions)
