import os
import pickle
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import DataParallel

import _functions as dep
import _models as models
import _preproc as preproc


def test_model(_model_name, test_dataset, test_num_each, _params, batch):
    num_test = len(test_dataset)
    test_useful_start_idx = dep.get_useful_start_idx(_params.get_sequence, test_num_each)

    num_test_we_use = len(test_useful_start_idx)

    test_we_use_start_idx = test_useful_start_idx[0:num_test_we_use]

    test_idx = []
    for i in range(num_test_we_use):
        for j in range(_params.get_sequence):
            test_idx.append(test_we_use_start_idx[i] + j)

    num_test_all = len(test_idx)

    dep.log('num test start idx : {:6d}'.format(len(test_useful_start_idx)))
    dep.log('last idx test start: {:6d}'.format(test_useful_start_idx[-1]))
    dep.log('num of test dataset: {:6d}'.format(num_test))
    dep.log('num of test we use : {:6d}'.format(num_test_we_use))
    dep.log('num of all test use: {:6d}'.format(num_test_all))

    test_loader = None
    model = None

    torch.cuda.empty_cache()
    test_loader = dep.data_loader(test_dataset, test_idx, batch, _params.get_workers)
    model = models.resnet_lstm()
    model = DataParallel(model)
    model.load_state_dict(torch.load("C:\\Users\\Tomi\\PycharmProjects\\phase_detection_laparo\\SV-RCNet\\out\\SV-RCNeT_epoch-25_length-4_batch-56_trainN-17_valN-2_testN-2_resize-24_test_6799.pkl"))

    if _params.get_use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss(size_average=False)

    model.eval()
    test_loss = 0.0
    test_corrects = 0
    test_start_time = time.time()

    all_preds = []
    prg_i = 0
    dep.log('START testing - {}'.format(time.strftime("%Y-%m-%d %H:%M:%S")))
    progressmax = (len(test_loader))


    for data in test_loader:
        inputs, labels_2 = data
        labels_2 = labels_2[(_params.get_sequence - 1)::_params.get_sequence]

        with torch.no_grad():
            inputs = Variable(inputs.cuda())  # inputs = Variable(inputs.cuda(), volatile=True)
            labels = Variable(labels_2.cuda())  # labels = Variable(labels_2.cuda(), volatile=True)
        try:
            outputs = model.forward(inputs, _params.get_sequence)
        except Exception as error:
            print(error)
            return 0
        outputs = outputs[_params.get_sequence - 1::_params.get_sequence]
        _, preds = torch.max(outputs.data, 1)

        for i in range(len(preds)):
            all_preds.append(preds[i])
        loss = criterion(outputs, labels)
        test_loss += loss.data
        test_corrects += torch.sum(preds == labels.data)

        prg_i += 1
        prg_s = prg_i / progressmax * 100
        el_time = time.time() - test_start_time
        rem_time = (100 - prg_s) * el_time / prg_s
        el_time_s = dep.timesplit(el_time)
        rem_time_s = dep.timesplit(rem_time)
        print(
            '\rStatus: {: 3.2f}%   Elapsed time: {}    Estimated remaining time: {}   GPU util: {}%   GPU mem: {}%'.format(
                prg_s,
                el_time_s,
                rem_time_s,
                torch.cuda.utilization(),
                torch.cuda.memory_usage()),
            end='')


    test_elapsed_time = time.time() - test_start_time
    test_accuracy = test_corrects / num_test_we_use
    test_average_loss = test_loss / num_test_we_use

    dep.log('leng of all preds: {}'.format(len(all_preds)))
    save_test = int("{:4.0f}".format(test_accuracy * 10000))
    model_pure_name, _ = os.path.splitext(_model_name)
    print(model_pure_name)
    pred_name = model_pure_name + '_test_' + str(save_test) + '.pkl'

    with open(pred_name, 'wb') as f:
        pickle.dump(all_preds, f)

    dep.log('test elapsed: {:2.0f}m{:2.0f}s'
            ' test loss: {:4.4f}'
            ' test accu: {:.4f}'
            .format(test_elapsed_time // 60,
                    test_elapsed_time % 60,
                    test_average_loss, test_accuracy))
    return 1


if __name__ == "__main__":
    params = dep.load_params()
    logfile = '{}\\{}'.format(params.get_log_dir, datetime.now().strftime("SV-RCNet_train_%Y%m%d_%H%M%S.log"))
    dep.logging.basicConfig(filename=logfile, level=dep.logging.INFO)

    dep.log('training videos    : {:6d}'.format(params.get_train_set))
    dep.log('validation videos  : {:6d}'.format(params.get_valid_set))
    dep.log('test videos        : {:6d}'.format(params.get_test_set))

    preproc.run(params.get_train_set, params.get_valid_set, params.get_test_set, params)
    dep.print_params(params)
    _, _, _, _, test_dataset, test_num_each = dep.get_data(
        '../data/train_val_test_paths_labels.pkl', params)
    model_name = "./out/" + input('Please enter the model name (*.pth filename):\n').split('.')[0]+'.pth'
    batch = params.get_test_batch
    suc = 0
    while suc == 0:
        suc = test_model(model_name, test_dataset, test_num_each, params, batch)
        nbatch = batch - int(max(batch * 0.1, 1) // 1)
        if suc == 0:
            print('Batch number ({}) was too high to fit in memory. Trying lower ({}) number.\n'.format(batch, nbatch))
            batch = nbatch
        else:
            break
        if batch == 0:
            print('Too large images to input into the network, please configure the resize parameters lower')
            break

