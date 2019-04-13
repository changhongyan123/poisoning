import numpy as np
import random
from utility import LossHistory
import keras
from utility import convert_dataset,LossHistory
import pickle
from keras.utils import to_categorical


def load_poisoning_data(file_name):
    file_name = "poisoningData/" + file_name + ".pkl"
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    poisoning_data = {}

    poisoning_data["X"] = np.array(data["X"])
    poisoning_data["Y"]  = np.array(data["Y"])

    return poisoning_data


def save_poisoning_data(file,poisoning_data):
    file = "poisoningData/" + file +".pkl"
    f = open(file,"wb")
    pickle.dump(poisoning_data,f)
    f.close()



def posioning_attack(type,dataset,model_structure,poisoning_fraction,training_info,is_load):
    #input:
    #type: the attacker algorithm
    #dataset: {
    #   "clean_train": clean training dataset
    #   "clean_test": clean test dataset
    #   }
    #model_structure: model of the victim attack
    #poisoning_fraction: generate how many poisoning points
    #output:
    #the posioned dataset
    clean_train = dataset["clean_train"]
    clean_test = dataset["clean_test"]

    x_train,y_train,x_test,y_test =convert_dataset(training_info['model_name'],clean_train["X"],clean_train["Y"],clean_test["X"],clean_test["Y"])


    if type == 'loss':
        posioning_data = loss_attack(x_train,y_train,x_test,y_test,poisoning_fraction,model_structure,is_load)
    elif type == 'lable attack':
        posioning_data = label_flip_attack(x_train,y_train,x_test,y_test,poisoning_fraction,is_load)
    elif type == 'random attack':
        posioning_data = noisy_attack(x_train,y_train,x_test,y_test,poisoning_fraction,is_load)
    elif type == 'gradient ascent':
        posioning_data = gradient_ascent_attack(x_train,training_info['dataset_name'],poisoning_fraction,is_load,n_round=training_info['posioned_round'])
    elif type == 'min max':
        posioning_data = min_max_attack(x_train,y_train,x_test,y_test,poisoning_fraction,model_structure,is_load)
    elif type == 'influence':
        posioning_data = influence_attack(clean_train,clean_test,poisoning_fraction,is_load)
    elif type == 'generative':
        posioning_data = generative_model(x_train,training_info['dataset_name'],poisoning_fraction,is_load,n_round=training_info['posioned_round'])

    dataset["poisoning_data"] = posioning_data
    return dataset

def label_flip_attack(x_train,y_train,x_test,y_test,fraction,is_load,num_classes=10):
    #Just change the label attack
    if is_load == False:
        sample = x_train.shape[0]
        poisoing_sample = int(sample * fraction)

        x_p = np.copy(x_train[:poisoing_sample])
        y_p = np.copy(y_train[:poisoing_sample])

        #generate random label

        labels_p = np.random.randint(num_classes, size=y_p.shape[0])
        y_p = to_categorical(labels_p, num_classes=num_classes)

        poisoning_data = {}
        poisoning_data["X"] = x_p
        poisoning_data["Y"] = y_p

        file_name = 'label_flip_' + str(fraction)
        save_poisoning_data(file_name,poisoning_data)
        print('finish generate the posioning data: lable attack')
    else:
        file_name = 'label_flip_' + str(fraction)
        poisoning_data = load_poisoning_data(file_name)
    return  poisoning_data




def loss_attack(x_train,y_train,x_test,y_test,fraction,model,is_load,n_built_in = 5):
    if is_load == False:
        # generate the posioning data based on the loss. each class repeat 10 times.

        model.fit(x_train,y_train,epochs=n_built_in,verbose=0)
        n_bad_points = int(fraction*x_train.shape[0])
        poisoning_point = {"X":[],"Y":[]}
        history = LossHistory()
        generate_points = int(n_bad_points/10)
        loss = model.evaluate(x_train,y_train,verbose = 0,callbacks=[history],batch_size=1)
        max_index = np.argsort(history.losses)[-generate_points:]
        x_i = x_train[max_index]
        y_i = y_train[max_index]

        x_i = np.repeat(x_i,10,axis=0)
        y_i = np.repeat(y_i,10,axis=0)
        # poisoning_point = {}
        poisoning_point["X"].append(x_i)
        poisoning_point["Y"].append(y_i)
        file_name = 'loss_attack_' + str(fraction)
        save_poisoning_data(file_name,poisoning_point)
        return poisoning_point
    else:
        file_name = 'loss_attack_' + str(fraction)
        poisoning_point = load_poisoning_data(file_name)
        return poisoning_point

def gradient_ascent_attack(x_train,datatype,fraction,is_load,n_round,num_classes=10):
    #n n_round is the round of the poisoning generate
    sample = x_train.shape[0]
    poisoing_sample = int(sample * fraction)
    if is_load ==True:
        filename = 'poisoningData/gradient_attack_'+datatype+'.npz'
        data = np.load(filename)

        x_p = data['X'][:,n_round]
        x_p = x_p.reshape(-1,data['X'].shape[-1])
        label_p = data['Y'][:,n_round].reshape(-1)
        y_p = to_categorical(label_p, num_classes=num_classes)

        n_repeat = int(poisoing_sample/x_p.shape[0])
        x_p = np.repeat(x_p,n_repeat,axis=0)
        y_p = np.repeat(y_p,n_repeat,axis=0)

        poisoning_data= {}
        poisoning_data["X"] = x_p
        poisoning_data["Y"]  = y_p
        return poisoning_data

def generative_model(x_train,datatype,fraction,is_load,n_round,num_classes=10):
    #n n_round is the round of the poisoning generate
    sample = x_train.shape[0]
    poisoing_sample = int(sample * fraction)
    if is_load ==True:
        filename = 'poisoningData/generative_attack_'+datatype+'_50.npz'
        data = np.load(filename)

        x_p = data['X'][:,n_round]
        x_p = x_p.reshape(-1,data['X'].shape[-1])
        label_p = data['Y'][:,n_round].reshape(-1)
        y_p = to_categorical(label_p, num_classes=num_classes)

        n_repeat = int(poisoing_sample/x_p.shape[0])
        print('repeat '+str(n_repeat)+'times')
        x_p = np.repeat(x_p,n_repeat,axis=0)
        y_p = np.repeat(y_p,n_repeat,axis=0)

        poisoning_data= {}
        poisoning_data["X"] = x_p
        poisoning_data["Y"]  = y_p
    return poisoning_data

def noisy_attack(x_train,y_train,x_test,y_test,fraction,is_load,num_classes=10):
    if is_load == False:
        sample = x_train.shape[0]
        poisoing_sample = int(sample * fraction)

        x_p = np.copy(x_train[:poisoing_sample])
        y_p = np.copy(y_train[:poisoing_sample])
        noisy = np.random.normal(0,1,x_p.shape)
        x_p = x_p + noisy
        y_p = to_categorical(x_p, num_classes=num_classes)

        poisoning_data = {}
        poisoning_data["X"] = x_p
        poisoning_data["Y"] = y_p

        file_name = 'nosiy_attack' + str(fraction)
        save_poisoning_data(file_name,poisoning_data)
        print('finish generate the posioning data: noisy attack')
    else:
        file_name = 'label_flip_' + str(fraction)
        poisoning_data = load_poisoning_data(file_name)
    return  poisoning_data
