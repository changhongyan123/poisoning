# mnist, generative method, gradient to output of autoencoder

import mxnet as mx
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import time
import copy


def save_poisoning_data(filename,x,y):
    np.savez(filename, X=x,Y=y)



def SGD(key, weight, grad, lr=0.01, grad_norm=1):
    # key is key for weight, we can customize update rule
    # weight is weight array
    # grad is grad array
    # lr is learning rate
    # grad_norm is scalar to norm gradient, usually it is batch_size
    norm = 1.0 / grad_norm
    # here we can bias' learning rate 2 times larger than weight
    if "weight" in key or "gamma" in key:
        weight[:] -= lr * (grad * norm)
    elif "bias" in key or "beta" in key:
        weight[:] -= 2.0 * lr * (grad * norm)
        # weight[:] -= lr * (grad * norm)
    else:
        pass

def CalLogLoss(pred_prob, label):
    loss = 0.
    for i in range(pred_prob.shape[0]):
        loss += -np.log(max(pred_prob[i, int(label[i])], 1e-10))
    return loss



def generate_poisoning_fun(index = 0,total_round=5):
    # load dataset, model structure, parameter
    dev = mx.gpu(0)
    batch_size = 1
    # data_shape = (batch_size, 2352)
    data_shape = (batch_size, 3, 28, 28)
    train_iter = mx.io.ImageRecordIter(
        path_imgrec="data/cifar10/cifar10_train.rec",
        data_shape=(3, 28, 28),
        batch_size=batch_size,
        # mean_img="data/cifar10/mean.bin",
        rand_crop=True,
        rand_mirror=True,
        round_batch=True)

    val_iter = mx.io.ImageRecordIter(
        path_imgrec="data/cifar10/cifar10_val.rec",
        data_shape=(3, 28, 28),
        batch_size=batch_size,
        # mean_img="data/cifar10/mean.bin",
        rand_crop=True,
        rand_mirror=True,
        round_batch=True)


    all_data=[]
    all_label=[]
    poisoning_d = []
    poisoning_l = []
    poisoing_data_list  = []
    poisoing_label_list = []

    for i in range(index):
        batch_data =  copy.deepcopy(train_iter.next())
        poisoning_d.append(batch_data.data[0])
        poisoning_l.append(batch_data.label[0]+1)

    for j in range(index):
        with mx.Context(dev):
            # load original model
            softmax, arg_params, aux_params = mx.model.load_checkpoint('model/cifar10_model', 300)
            model = mx.mod.Module(softmax, context=dev)
            model.bind(data_shapes=train_iter.provide_data,
                       label_shapes=train_iter.provide_label,
                       inputs_need_grad=True)
            model.set_params(arg_params, aux_params)

            # define autoencoder
            de_out = mx.symbol.load('model/cifar10_model-symbol.json')
            ae_arg_arrays_load = mx.nd.load('model/cifar10_model-0300.params')


            ae_arg_shapes, ae_output_shapes, ae_aux_shapes = de_out.infer_shape(data=data_shape)
            ae_grad_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in ae_arg_shapes]
            ae_arg_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in ae_arg_shapes]

            ae_model = de_out.simple_bind(ctx=dev, data=(1, batch_size,28,28), grad_req='write')

            # load pre-trained weight
            print(len(ae_arg_arrays_load))
            for i in range(1, len(ae_arg_arrays_load)):
                ae_arg_arrays_load[i].copyto(ae_model.arg_arrays[i])

            train_iter.reset()
            dataBatchP = copy.deepcopy(train_iter.next())
            poisoning_d[j].copyto(dataBatchP.data[0])
            poisoning_l[j].copyto(dataBatchP.label[0])
            data = dataBatchP.data[0]

            num_normal = 1000
            attacked_model_lr = 0.005
            generative_model_lr = 0.001
            model.init_optimizer(kvstore='local', optimizer='sgd', optimizer_params=(('learning_rate', attacked_model_lr),))
            # get normal data loss and accuracy
            loss = 0
            for num in range(num_normal):
                dataBatch = copy.deepcopy(train_iter.next())
                label = dataBatch.label[0]
                model.forward(dataBatch)
                output = model.get_outputs()[0].asnumpy()
                loss += CalLogLoss(output, label.asnumpy())
            print 'normal data loss: %.4f' % loss

            val_iter.reset()
            metric = mx.metric.create('acc')
            for batch in val_iter:
                model.forward(batch, is_train=False)
                model.update_metric(metric, batch.label)
            print metric.get()
            # val_iter.reset()
            # val_acc = model.score(val_iter, 'acc')
            # print 'Val Acc: %.4f' % val_acc[0][1]

            #attack with initial data, get normal data loss and accuracy
            print 'after initial attack'
            model.forward(dataBatchP, is_train=True)
            model.backward()
            model.update()
            val_iter.reset()
            metric.reset()
            for batch in val_iter:
                model.forward(batch, is_train=False)
                model.update_metric(metric, batch.label)
            print metric.get()
            # val_iter.reset()
            # val_acc = model.score(val_iter, 'acc')
            # print 'Val Acc: %.4f' % val_acc[0][1]
            # re-evaluate normal data loss
            loss = 0
            train_iter.reset()
            dataBatch = copy.deepcopy(train_iter.next())
            for num in range(num_normal):
                dataBatch = copy.deepcopy(train_iter.next())
                label = dataBatch.label[0]
                model.forward(dataBatch)
                loss += CalLogLoss(model.get_outputs()[0].asnumpy(), label.asnumpy())
            print 'normal data loss: %.4f' % loss
            plt.figure('poisoned data')
            plt.subplot(1, 5, 1)
            plt.imshow((dataBatchP.data[0]).asnumpy()[0].astype(np.uint8).transpose(1, 2, 0))
            # generate poisoned data
            ae_de_grad = mx.nd.zeros(ae_model.outputs[0].shape, ctx=dev)
            pre_loss = 0
            for round in range(total_round):
                start = time.time()
                print 'round %d' % round
                # update original model
                train_iter.reset()
                dataBatch = copy.deepcopy(train_iter.next())

                ae_model.arg_dict['data'][:] = dataBatchP.data[0].reshape((1,2352))/255
                ae_model.forward()
                ae_output = ae_model.outputs[0].asnumpy()
                label_tmp = copy.deepcopy(dataBatchP.label)
                dataBatch_tmp = copy.deepcopy(mx.io.DataBatch([mx.nd.array(ae_output.reshape(data_shape))], label_tmp))
                # load pre-trained weight
                model.set_params(arg_params, aux_params)
                model.forward(dataBatch_tmp, is_train=True)
                model.backward()
                # update attacked model
                model.update()
                print 'poisoned network'

                val_iter.reset()
                metric.reset()
                for batch in val_iter:
                    model.forward(batch, is_train=False)
                    model.update_metric(metric, batch.label)
                print metric.get()
                # val_iter.reset()
                # val_acc = model.score(val_iter, 'acc')
                # print 'Val Acc: %.4f' % val_acc[0][1]

                if round%2 == 0:
                    plt.subplot(1, 5, round/2+2)
                    plt.imshow((ae_output.reshape(3, 28, 28) * 255).astype(np.uint8).transpose(1, 2, 0))

                # get normal data loss
                loss = 0
                tmpGrad = np.zeros(data.shape)
                dataBatch = copy.deepcopy(train_iter.next())
                for num in range(num_normal):
                    dataBatch = copy.deepcopy(train_iter.next())
                    label = dataBatch.label[0]
                    model.forward(dataBatch, is_train=True)
                    model.backward()
                    output = model.get_outputs()[0].asnumpy()
                    loss += CalLogLoss(output, label.asnumpy())
                    tmpGrad += model.get_input_grads()[0].asnumpy()
                # ae_de_grad[:] = -np.sign(tmpGrad.reshape(1,2352))*1
                ae_de_grad[:] = -np.sign(tmpGrad.reshape(1,2352)) * np.sign(loss - pre_loss)
                ae_model.backward([ae_de_grad])
                for key in ae_model.arg_dict.keys():
                    SGD(key, ae_model.arg_dict[key], ae_model.grad_dict[key], generative_model_lr,batch_size)
                end = time.time()
                print 'time: %.4f' % (end-start)
                print 'Update autocoder'
                print 'normal data loss: %.4f' % loss
                pre_loss = loss
                poisoing_data_list.append(ae_output)
                poisoing_label_list.append(dataBatchP.label[0].asnumpy())
            all_data.append(poisoing_data_list)
            all_label.append(poisoing_label_list)
    return all_data,all_label


data,label = generate_poisoning_fun(index=2,total_round=3)
filename = '../poisoningData/generative_attack_cifar.npz'
save_poisoning_data(filename,data,label)
