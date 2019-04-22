import keras
import numpy as np
from keras.utils import to_categorical
from keras.models import model_from_json
import matplotlib.pyplot as plt

class LossHistory(keras.callbacks.Callback):
    def on_test_begin(self, logs={}):
        self.losses = []

    def on_test_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def save_loss(filename,x,y):
    np.savez(filename, clean_loss=x,poisoning_loss=y)


def save_model(model,training_info):
    if training_info['is_poisoning'] == True:
        name =  training_info['training_type'] + training_info['dataset_name'] + training_info['model_name'] + training_info['poisoning_type'] + str(training_info['poisoning_fraction'])
        weight_file = 'model/weight/attack_' + name + '.h5'
        model_file =  'model/model_structure/attack_' + name + '.json'
    else:
        name =  training_info['training_type'] + training_info['dataset_name'] + training_info['model_name']
        weight_file = 'model/weight/clean_' + name + '.h5'
        model_file =  'model/model_structure/clean_' + name + '.json'

    model.save_weights(weight_file)

    with open(model_file, 'w') as f:
        f.write(model.to_json())

    print('save model successfully')

def load_model(weight_file,model_file):
    with open(model_file, 'r') as f:
        model = model_from_json(f.read())
    # Load weights into the new model
    model.load_weights(weight_file)
    return model

def save_history(history,filename):
    acc_path = 'figure/acc/'
    loss_path = 'figure/loss/'
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(acc_path+filename+'_acc.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(loss_path+filename+'_loss.png')
    plt.clf()


def convert_dataset(type,x_train,y_train,x_test,y_test,training_info):
    input_dim = x_test[0].shape
    output_dim = len(np.unique(y_train))
    if training_info['training_type'] != 'classification':
        y_test = to_categorical(y_test, num_classes=output_dim)
        y_train = to_categorical(y_train, num_classes=output_dim)

    #for the image input
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if type != 'CNN':
        input_dim = x_test[0].size

        x_train = x_train.reshape(-1,input_dim)
        output_dim = len(np.unique(y_train))
        x_test = x_test.reshape(-1,input_dim)

    return x_train,y_train,x_test,y_test
