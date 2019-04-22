#!/usr/bin/python
# -*- coding: UTF-8 -*-
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
from utility import LossHistory,save_model

import keras
import pickle
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.models import load_model

# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2"


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset_list = ['mnist','fashion mnist','cifar10','cifar100']
attack_list = ['gradient_ascent','generative','lable_attack','random_attack']#,
# attack_list = ['lable_attack']#



if __name__ == '__main__':



    training_info = {}
    training_info['is_poisoning'] = True
    training_info['training_epoch'] = 1
    training_info['learning_rate'] = 0.01
    training_info['dataset_name'] = 'mnist'
    training_info['model_name'] = 'MLP' #'CNN'
    training_info['batch_size'] = 256
    training_info['print_model'] = False
    training_info['print_process'] = 0
    training_info['plot_loss'] = 0
    training_info['training_type'] = 'normal'#'normal'#'transfer','normal'
    dataset = load_dataset(training_info['dataset_name'])


    parameters = {}
    if training_info['model_name'] == 'CNN':
        parameters['input_shape'] = dataset['clean_train']['X'][0].shape
    else:
        parameters['input_shape'] = dataset['clean_train']['X'][0].size
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
    functions['optimizer'] = 'adam'
    functions['loss'] = 'categorical_crossentropy'
    functions['activation'] = 'softmax'
    functions['metrics'] = ['accuracy']
    metrics = ['accuracy']

    all_history  = []



    for training_method in ['online']:
        for f in [0.2]:
            training_info['poisoning_fraction'] = f
            training_info['is_poisoning'] = False
            training_info['training_type'] = training_method

            model = create_model(training_info['model_name'],parameters,functions,metrics)
            model,history = call_training(model,dataset,training_info,parameters,functions)
            # save_model(model,training_info)
            history['poisoning_type'] = 'None'
            history['training_type'] = training_info['training_type']
            # all_history.append(history)

            training_info['is_poisoning'] = True
            for attack in attack_list:
                training_info['poisoning_type'] = attack
                if attack == 'gradient_ascent':
                    training_info['posioned_round'] = 4
                elif attack == 'generative':
                    training_info['posioned_round'] = 9
                model_p = create_model(training_info['model_name'],parameters,functions,metrics)
                model_p,history = call_training(model_p,dataset,training_info,parameters,functions)
                history['poisoning_type'] = training_info['poisoning_type']
                history['fraction'] = training_info['poisoning_fraction']
                history['training_type'] = training_info['training_type']
        #         all_history.append(history)
        #         # save_model(model_p,training_info)
        # with open("history"+training_method+".txt", "wb") as fp:   #Pickling
        #     pickle.dump(all_history, fp)
