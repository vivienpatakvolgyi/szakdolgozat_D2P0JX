import copy
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import DataParallel

import _functions as dep
import _models as models
import _preproc as preproc


def train_model(train_dataset, train_num_each, val_dataset, val_num_each, _params, trbatch, vbatch):
    num_train = len(train_dataset)
    num_val = len(val_dataset)

    train_useful_start_idx = dep.get_useful_start_idx(_params.get_sequence, train_num_each)
    val_useful_start_idx = dep.get_useful_start_idx(_params.get_sequence, val_num_each)

    num_train_we_use = len(train_useful_start_idx) // _params.get_num_gpu * _params.get_num_gpu
    num_val_we_use = len(val_useful_start_idx) // _params.get_num_gpu * _params.get_num_gpu

    train_we_use_start_idx = train_useful_start_idx[0:num_train_we_use]
    val_we_use_start_idx = val_useful_start_idx[0:num_val_we_use]

    train_idx = []
    for i in range(num_train_we_use):
        for j in range(_params.get_sequence):
            train_idx.append(train_we_use_start_idx[i] + j)

    val_idx = []
    for i in range(num_val_we_use):
        for j in range(_params.get_sequence):
            val_idx.append(val_we_use_start_idx[i] + j)

    num_train_all = len(train_idx)
    num_val_all = len(val_idx)
    dep.log('num of train dataset: {:6d}'.format(num_train))
    dep.log('num train start idx : {:6d}'.format(len(train_useful_start_idx)))
    dep.log('last idx train start: {:6d}'.format(train_useful_start_idx[-1]))
    dep.log('num of train we use : {:6d}'.format(num_train_we_use))
    dep.log('num of all train use: {:6d}'.format(num_train_all))
    dep.log('num of valid dataset: {:6d}'.format(num_val))
    dep.log('num valid start idx : {:6d}'.format(len(val_useful_start_idx)))
    dep.log('last idx valid start: {:6d}'.format(val_useful_start_idx[-1]))
    dep.log('num of valid we use : {:6d}'.format(num_val_we_use))
    dep.log('num of all valid use: {:6d}'.format(num_val_all))

    train_loader = dep.data_loader(train_dataset, train_idx, trbatch, _params.get_workers)
    val_loader = dep.data_loader(val_dataset, val_idx, vbatch, _params.get_workers)

    model = models.resnet_lstm()

    if _params.get_use_gpu:
        model = DataParallel(model)
        model.to(device)

    #if _params.get_use_gpu:
       # model = model.cuda()

    model = DataParallel(model)
    criterion = nn.CrossEntropyLoss(size_average=False)

    optimizer = optim.Adam(model.parameters(), lr=_params.get_lr)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_accuracy = 0.0
    correspond_train_acc = 0.0

    record_np = np.zeros([_params.get_epochs, 4])

    dep.log('')
    start_time = time.time()
    dep.log('START training - {}\n'.format(time.strftime("%Y-%m-%d %H:%M:%S")))
    progressmax = _params.get_epochs * (len(train_loader) + len(val_loader))
    train_loader = None
    torch.cuda.empty_cache()
    prg_i = 0
    for epoch in range(_params.get_epochs):
        # np.random.seed(epoch)
        torch.cuda.empty_cache()
        np.random.shuffle(train_we_use_start_idx)
        train_idx = []
        for i in range(num_train_we_use):
            for j in range(_params.get_sequence):
                train_idx.append(train_we_use_start_idx[i] + j)

        train_loader = dep.data_loader(train_dataset, train_idx, trbatch, _params.get_workers)
        model.train()

        train_loss = 0.0
        train_corrects = 0
        train_start_time = time.time()
        for data in train_loader:
            inputs, labels_2 = data
            if _params.get_use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels_2.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels_2)

            optimizer.zero_grad()
            try:
                if torch.cuda.memory_reserved() > 0.8 * 8 * 1024 * 1024 * 1024:
                    print(torch.cuda.memory_reserved()//(1024 * 1024 * 1024))
                    return 0
                outputs = model.forward(inputs, _params.get_sequence)
            except Exception as error:
                print(error)
                return 0
            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            try:
                if torch.cuda.memory_reserved() > 0.8 * 8 * 1024 * 1024 * 1024:
                    print(torch.cuda.memory_reserved()//(1024 * 1024 * 1024))
                    return 0
                loss.backward()
            except Exception as error:
                print(error)
                return 0
            optimizer.step()
            train_loss += loss.data
            train_corrects += torch.sum(preds == labels.data)

            prg_i += 1
            prg_s = prg_i / progressmax * 100
            el_time, el_time_s, rem_time_s, _ = dep.run_stats(prg_s, start_time)
            if epoch == 0:
                rem_time_s = '0:00:00'
            stats = '\rStatus: {:4.2f}% (training)  Elapsed time: {:8s}  Est time: {:8s}  GPU util: {:3d}%  GPU mem: {:3.1f}% '.format(prg_s, el_time_s, rem_time_s, torch.cuda.utilization(),
                                                                                                                                     torch.cuda.memory_reserved() / (8 * 1024 * 1024 * 1024)*100)
            print(stats, end=' ')

        train_elapsed_time = time.time() - train_start_time
        train_accuracy = train_corrects / num_train_all
        train_average_loss = train_loss / num_train_all

        # begin eval
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_start_time = time.time()
        for data in val_loader:
            inputs, labels_2 = data
            labels_2 = labels_2[(_params.get_sequence - 1)::_params.get_sequence]
            if _params.get_use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels_2.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels_2)

            try:
                if torch.cuda.memory_reserved() > 0.8 * 8 * 1024 * 1024 * 1024:
                    print(torch.cuda.memory_reserved() // (1024 * 1024 * 1024))
                    return -1
                outputs = model.forward(inputs, _params.get_sequence)
            except Exception as error:
                print(error)
                return -1

            outputs = outputs[_params.get_sequence - 1::_params.get_sequence]

            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            val_loss += loss.data
            val_corrects += torch.sum(preds == labels.data)

            prg_i += 1
            prg_s = prg_i / progressmax * 100
            el_time, el_time_s, rem_time_s, _ = dep.run_stats(prg_s, start_time)
            if epoch == 0:
                rem_time_s = '0:00:00'
            stats = '\rStatus: {:4.2f}% (validating)  Elapsed time: {:8s}  Est time: {:8s}  GPU util: {:3d}%  GPU mem: {:3.1f}% '.format(prg_s, el_time_s, rem_time_s, torch.cuda.utilization(),
                                                                                                                                     torch.cuda.memory_reserved() / (8 * 1024 * 1024 * 1024)*100)
            print(stats, end=' ')
        val_elapsed_time = time.time() - val_start_time
        val_accuracy = val_corrects / num_val_we_use
        val_average_loss = val_loss / num_val_we_use
        el_time, el_time_s, rem_time_s, speed = dep.run_stats(prg_s, start_time)
        dep.log(
            '\rEpoch:  {:3d} /{:3d}   Elapsed time: {}    Estimated remaining time: {} '.format(epoch,
                                                                                                _params.get_epochs,
                                                                                                el_time_s,
                                                                                                rem_time_s))
        dep.log(' train in: {:2.0f}m{:2.0f}s'
                ' train loss: {:4.4f}'
                ' train accu: {:.4f}'.format(train_elapsed_time // 60, train_elapsed_time % 60, train_average_loss,
                                             train_accuracy))
        dep.log(' valid in: {:2.0f}m{:2.0f}s'
                ' valid loss: {:4.4f}'
                ' valid accu: {:.4f}'.format(val_elapsed_time // 60, val_elapsed_time % 60, val_average_loss,
                                             val_accuracy))

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            correspond_train_acc = train_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
        if val_accuracy == best_val_accuracy:
            if train_accuracy > correspond_train_acc:
                correspond_train_acc = train_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())

        record_np[epoch, 0] = train_accuracy
        record_np[epoch, 1] = train_average_loss
        record_np[epoch, 2] = val_accuracy
        record_np[epoch, 3] = val_average_loss

    dep.log('best accuracy: {:.4f} cor train accu: {:.4f}'.format(best_val_accuracy, correspond_train_acc))
    dep.log(model)
    save_val = int("{:4.0f}".format(best_val_accuracy * 10000))
    save_train = int("{:4.0f}".format(correspond_train_acc * 10000))

    name = ("out\\SV-RCNeT" +
            "_epoch-" + str(_params.get_epochs) +
            "_length-" + str(_params.get_sequence) +
            "_batch-" + str(_params.get_train_batch) +
            # "_train-" + str(save_train) +
            # "_val-" + str(save_val) +
            "_trainN-" + str(params.get_train_set) +
            "_valN-" + str(params.get_valid_set) +
            "_testN-" + str(params.get_test_set) +
            "_resize-" + str(int(_params.get_resize * _params.get_train_resize * 100)))

    model_name = name + ".pth"
    torch.save(best_model_wts, model_name)

    record_name = name + ".npy"
    np.save(record_name, record_np)

    model_dict_file = name + ".model"
    print(model, file=open(model_dict_file, 'a'))
    return 1


if __name__ == "__main__":
    params = dep.load_params()
    logfile = '{}\\{}'.format(params.get_log_dir, datetime.now().strftime("SV-RCNet_train_%Y%m%d_%H%M%S.log"))
    dep.logging.basicConfig(filename=logfile, level=dep.logging.INFO)

    device = torch.device("cuda:0" if params.get_use_gpu else "cpu")

    dep.log('training videos    : {:6d}'.format(params.get_train_set))
    dep.log('validation videos  : {:6d}'.format(params.get_valid_set))
    dep.log('test videos        : {:6d}'.format(params.get_test_set))

    preproc.run(params.get_train_set, params.get_valid_set, params.get_test_set, params)
    dep.print_params(params)
    train_dataset, train_num_each, val_dataset, val_num_each, _, _ = dep.get_data(
        '../data/train_val_test_paths_labels.pkl', params)

    trbatch = params.get_train_batch
    vbatch = params.get_valid_batch

    suc = 0
    while suc != 1:
        suc = train_model(train_dataset, train_num_each, val_dataset, val_num_each, params, trbatch, vbatch)
        ntrbatch = trbatch - int(min((trbatch) / 2, 8) // 1)
        nvbatch = vbatch - int(max(vbatch * 0.1, 1) // 1)

        if suc == 0:
            print('\nBatch number ({}) was too high to fit in memory. Trying lower ({}) number.\n'.format(trbatch,
                                                                                                        ntrbatch))
            trbatch = ntrbatch
        elif suc == -1:
            print(
                '\nBatch number ({}) was too high to fit in memory. Trying lower ({}) number.\n'.format(vbatch, nvbatch))
            vbatch = nvbatch
        else:
            break

        if trbatch < 16:
            print(
                '\nToo large images for training to input into the network, please configure the resize parameters lower')
            break
        elif vbatch < 1:
            print(
                '\nToo large images for validation to input into the network, please configure the resize parameters lower')
            break
