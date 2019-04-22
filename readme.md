# The vulnerability of the neural network under data poisoning attacks

This project is to analysis the nerual network under the posioning attack from the feature learning aspect.

Now, the code already contain the transfer learning, online learning and the end to end learning settings. For the dataset, it supprt the MNIST dataset and CIFAR10 dataset.


### Gnenerate the optimal poisoning data.

This part is from the previous work: [Generative Poisoning Attack Method Against Neural Networks](<https://github.com/yangcf10/Poisoning-attack/>)

It contains the generative model attack and gradient model attack.

- Dataset :

  Training and testing data for MNIST should be saved at: image="optimal_attack/data/mnist/train-images-idx3-ubyte" label="optimal_attack/data/mnist/train-labels-idx1-ubyte" image="data/mnist/t10k-images-idx3-ubyte" label="optimal_attack/data/mnist/t10k-labels-idx1-ubyte"

  Training and testing data for Cifar-10 should be saved at: path_imgrec="optimal_attack/data/cifar10/cifar10_train.rec" path_imgrec="optimal_attack/data/cifar10/cifar10_val.rec"

- Environment:

  python 2.7

  mxnet

  matplotlib

- Run:

  mnist_direct.py: Direct gradient method on MNIST. 

  mnist_generative.py: Generative gradient method on MNIST. 

  cifar_direct.py: Direct gradient method on Cifar-10. 

  cifar_generative.py: Generative gradient method on Cifar-10.

  In the file, modify the the last three lines:

  ```python
  # index is how many poisoned data you want to generate.
  # total_round is how many steps you want to update your poisoned data 
  data,label = generate_poisoning_fun(index=600,total_round=5)
  # the poisoned data will be saved as filename. 
  filename = '../poisoningData/gradient_attack_cifar.npz'
  # call the generation process
  save_poisoning_data(filename,data,label)
  ```

- Result:

  The generated poisoned data will be saved. To use in the analysis, the data should be saved as the default name.

### Analysis the neural network

To analysis the performance of the neural network under the posioning attack.In the project, we evaluate the nerual network in the transfer learning settings, end to end training setting and online learning settings.

- Dataset:

  The dataset is from keras dataset. It contains MNIST,MNIST Fashion,Cifar10,Cifar100.

  The optimal attack only support MNIST and Cifar10. You can run with label flip attack ,random nosie attack.

- Environment:

  python 3.6

  tensorflow

  scikit-learn

  Keras

- Run: 

  Run main.py.

  Modify the training_info,parameters,functions as your requirments.

  ```
  training_info['is_poisoning'] = True    # trained the clean dataset or poisoned dataset  
  training_info['training_epoch'] = 1     # training epoches 
  training_info['learning_rate'] = 0.01   # learning rate
  training_info['dataset_name'] = 'mnist' # dataset
  training_info['model_name'] = 'MLP'     #'CNN'
  training_info['batch_size'] = 256       # batch size 
  training_info['print_model'] = False    # whether to print the victim model
  training_info['print_process'] = 0      # whether print the training information
  training_info['plot_loss'] = 0			# whether plot the training loss
  training_info['training_type'] = 'normal' #['normal','online','normal'] decide the training setting
  
  
  training_info['poisoning_type']          # the attack type
  
  
  parameters['output_shape'] = len(np.unique(dataset['clean_train']['Y'])) # class number
  parameters['fc_layer']['kernel_init'] = 'random_uniform' # init method
  parameters['fc_layer']['bias_init'] = 'ones'			 # init method
  
  parameters['hidden_layer'] = [64,10]					# MLP parameters
  parameters['fc_layer'] = [256,128,10]					# CNN parameter
  parameters['cnn_layer'] = {}
  parameters['cnn_layer']['kernel_init'] = 'glorot_uniform' # cnn filter init method
  parameters['cnn_layer']['bias_init'] = 'zeros'            # cnn filter init method
  parameters['cnn_layer']['padding'] = 'SAME'               # padding method for cnn 
  parameters['cnn_layer']['filter_number'] = [18,18]        # how manys filters in each layer
  parameters['cnn_layer']['filter_shape'] =[(14,14),(7,7)]  # filter size for each layer
  parameters['cnn_layer']['pooling_shape'] = [(2,2),(2,2)]  # pooling size of the model
  parameters['data_format'] = 'channels_last'
  
  
  functions['optimizer'] = 'adam'
  functions['loss'] = 'categorical_crossentropy'
  functions['activation'] = 'softmax'
  functions['metrics'] = ['accuracy']
  
  ```

