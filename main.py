from attack import posioning_attack
from model import create_model
from load_data import load_dataset

from train import convert_dataset
dataset_list = ['mnist','fashion mnist','cifar10','cifar100']



if __name__ == '__main__':
    poisoning_fraction = 0.2
    poisoning_type = 'lable attack'
    training_epoch = 50
    learning_rate = 0.1
    dataset_name = 'cifar10'
    model_name = 'CNN'
    batch_size = 128
    learning_epoch = 1

    dataset = load_dataset(dataset_name)
    x_train,y_train,x_test,y_test,input_dim,out_dim = convert_dataset(model_name,dataset)


    parameters = {}
    functions = {}
    functions['optimizer'] = 'adam'
    functions['loss'] = 'categorical_crossentropy'
    functions['activation'] = 'softmax'
    metrics = ['accuracy']
    parameters['input_shape'] = input_dim
    parameters['output_shape'] = out_dim


    # parameters['hidden_layer'] = [20,10,10] MLP parameters

    parameters['fc_layer'] = [20,10]
    parameters['cnn_layer'] = {}
    parameters['cnn_layer']['filter_number'] = [16,8,8]
    parameters['cnn_layer']['filter_shape'] =[(3,3),(3,3),(3,3)]
    parameters['cnn_layer']['pooling_shape'] = [(2,2),(2,2),(2,2)]
    parameters['data_format'] = 'channels_last'


    model = create_model(model_name,parameters,functions,metrics)
    model.fit(x_train,y_train, epochs=learning_epoch, batch_size=batch_size)
    score = model.evaluate(x_test, y_test, batch_size=batch_size)


    print('--------------Training Information---------------')
    print(model.summary())
    print('Input shape: '+ str(input_dim))
    print('Output shape: '+ str(out_dim))
    print('dataset: ' + dataset_name)
    print('model name: ' + model_name)
    print('learning_rate: '+ str(learning_rate))
    print('batch_size: '+ str(batch_size))
    print('learning epochs: '+ str(learning_epoch))
    print('evaluation :' + str(score))
    print('optimizer: ' + functions['optimizer'] )
    print('-------------------------------------------------')
