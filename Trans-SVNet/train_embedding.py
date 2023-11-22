# some code adapted from https://github.com/YuemingJin/MTRCNet-CL
# and https://github.com/YuemingJin/TMRNet
import argparse
import copy
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchinfo import summary

import _functions
import _models
import _transforms
from _functions import log

parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument('-g', '--gpu', default=True, type=bool, help='gpu use, default True')
parser.add_argument('-s', '--seq', default=1, type=int, help='sequence length, default 10')
parser.add_argument('-t', '--train', default=56, type=int, help='train batch size, default 400')
parser.add_argument('-v', '--val', default=10, type=int, help='valid batch size, default 10')
parser.add_argument('-o', '--opt', default=1, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=20, type=int, help='epochs to train and val, default 25')
parser.add_argument('-w', '--work', default=8, type=int, help='num of workers to use, default 4')
parser.add_argument('-l', '--lr', default=5e-4, type=float, help='learning rate for optimizer, default 5e-5')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
parser.add_argument('--weightdecay', default=5e-4, type=float, help='weight decay for sgd, default 0')
parser.add_argument('--dampening', default=0, type=float, help='dampening for sgd, default 0')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum, default False')
parser.add_argument('--sgdadjust', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
parser.add_argument('--sgdstep', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
parser.add_argument('--sgdgamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')

args = parser.parse_args()


def print_params():
    log('number of gpu   : {:6d}'.format(num_gpu))
    log('sequence length : {:6d}'.format(sequence_length))
    log('train batch size: {:6d}'.format(train_batch_size))
    log('valid batch size: {:6d}'.format(val_batch_size))
    log('optimizer choice: {:6d}'.format(optimizer_choice))
    log('multiple optim  : {:6d}'.format(multi_optim))
    log('num of epochs   : {:6d}'.format(epochs))
    log('num of workers  : {:6d}'.format(workers))
    log('learning rate   : {:.4f}'.format(learning_rate))
    log('momentum for sgd: {:.4f}'.format(momentum))
    log('weight decay    : {:.4f}'.format(weight_decay))
    log('dampening       : {:.4f}'.format(dampening))
    log('use nesterov    : {:6d}'.format(use_nesterov))
    log('method for sgd  : {:6d}'.format(sgd_adjust_lr))
    log('step for sgd    : {:6d}'.format(sgd_step))
    log('gamma for sgd   : {:.4f}'.format(sgd_gamma))


def train_model(train_dataset, train_num_each, val_dataset, val_num_each):
    print_params()
    ((train_dataset_80),
     (train_num_each_80),
     (val_dataset, test_dataset),
     (val_num_each, test_num_each)) = train_dataset, train_num_each, val_dataset, val_num_each

    train_useful_start_idx_80 = _functions.get_useful_start_idx(sequence_length, train_num_each_80)
    val_useful_start_idx = _functions.get_useful_start_idx(sequence_length, val_num_each)

    num_train_we_use_80 = len(train_useful_start_idx_80)
    num_val_we_use = len(val_useful_start_idx)

    train_we_use_start_idx_80 = train_useful_start_idx_80
    val_we_use_start_idx = val_useful_start_idx

    train_idx = []
    for i in range(num_train_we_use_80):
        for j in range(sequence_length):
            train_idx.append(train_we_use_start_idx_80[i] + j)

    val_idx = []
    for i in range(num_val_we_use):
        for j in range(sequence_length):
            val_idx.append(val_we_use_start_idx[i] + j)

    num_train_all = len(train_idx)
    num_val_all = len(val_idx)

    print('num train start idx 80: {:6d}'.format(len(train_useful_start_idx_80)))
    print('num of all train use: {:6d}'.format(num_train_all))
    print('num of all valid use: {:6d}'.format(num_val_all))

    train_idx_80 = []
    for i in range(num_train_we_use_80):
        for j in range(sequence_length):
            train_idx_80.append(train_we_use_start_idx_80[i] + j)

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=_transforms.SeqSampler(val_dataset, val_idx),
        num_workers=workers,
        pin_memory=False
    )
    train_loader = DataLoader(
        train_dataset_80,
        batch_size=train_batch_size,
        sampler=_transforms.SeqSampler(train_dataset_80, train_idx_80),
        num_workers=workers,
        pin_memory=False
    )

    log('')
    start_time = time.time()
    log('START training - {}\n'.format(time.strftime("%Y-%m-%d %H:%M:%S")))
    model = _models.resnet_lstm()

    if use_gpu:
        model = DataParallel(model)
        model.to(device)

    criterion_phase = nn.CrossEntropyLoss(reduction='sum')
    optimizer = None
    exp_lr_scheduler = None

    if multi_optim == 0:
        if optimizer_choice == 0:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=dampening,
                                  weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif multi_optim == 1:
        if optimizer_choice == 0:
            optimizer = optim.SGD([
                {'params': model.module.share.parameters()},
                {'params': model.module.fc.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10, momentum=momentum, dampening=dampening,
                weight_decay=weight_decay, nesterov=use_nesterov)
            if sgd_adjust_lr == 0:
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_adjust_lr, gamma=sgd_gamma)
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam([
                {'params': model.module.share.parameters()},
                {'params': model.module.fc.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10)

    best_model_wts = copy.deepcopy(model.module.state_dict())
    best_val_accuracy_phase = 0.0
    correspond_train_acc_phase = 0.0
    best_epoch = 0
    progressmax = epochs * (len(train_loader) + len(val_loader))
    train_loader = None
    torch.cuda.empty_cache()
    prg_i = 0
    throw = 0

    from torchinfo import summary
    log(summary(model, verbose=0))
    log('\n\n')
    input('waithere')

    for epoch in range(epochs):
        throw += 1
        torch.cuda.empty_cache()
        np.random.shuffle(train_we_use_start_idx_80)
        train_idx_80 = []
        for i in range(num_train_we_use_80):
            for j in range(sequence_length):
                train_idx_80.append(train_we_use_start_idx_80[i] + j)

        train_loader = DataLoader(
            train_dataset_80,
            batch_size=train_batch_size,
            sampler=_transforms.SeqSampler(train_dataset_80, train_idx_80),
            num_workers=workers,
            pin_memory=False
        )

        # Sets the module in training mode.
        model.train()
        train_loss_phase = 0.0
        train_corrects_phase = 0
        batch_progress = 0.0
        train_start_time = time.time()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            if use_gpu:
                inputs, labels_phase = data[0].to(device), data[1].to(device)
                if torch.cuda.memory_reserved() > 0.8 * 8 * 1024 * 1024 * 1024:
                    print(torch.cuda.memory_reserved() // (1024 * 1024 * 1024))
                    return -1
            else:
                inputs, labels_phase = data[0], data[1]

            labels_phase = labels_phase[(sequence_length - 1)::sequence_length]

            inputs = inputs.view(-1, sequence_length, 3, 224, 224)
            outputs_phase = model.forward(inputs)
            outputs_phase = outputs_phase[sequence_length - 1::sequence_length]

            _, preds_phase = torch.max(outputs_phase.data, 1)
            loss_phase = criterion_phase(outputs_phase, labels_phase)

            loss = loss_phase
            loss.backward()
            optimizer.step()

            train_loss_phase += loss_phase.data.item()

            batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
            train_corrects_phase += batch_corrects_phase

            prg_i += 1
            batch_progress += 1
            prg_s = prg_i / progressmax * 100
            el_time, el_time_s, _, _ = _functions.run_stats(prg_s, start_time)
            percent = batch_progress * train_batch_size / num_train_all * 100
            if epoch == 0:
                rem_time_s = '0:00:00'
            stats = '\rStatus: {:4.2f}% (training)  Epoch: {:4.2f}%  Elapsed time: {:8s}  Est time: {:8s} '.format(
                prg_s, min(100, percent), el_time_s, rem_time_s)
            print(stats, end=' ')

        train_elapsed_time = time.time() - train_start_time
        train_accuracy_phase = float(train_corrects_phase) / float(num_train_all) * sequence_length
        train_average_loss_phase = train_loss_phase / num_train_all * sequence_length

        # Sets the model in evaluation mode.
        model.eval()
        val_loss_phase = 0.0
        val_corrects_phase = 0
        val_start_time = time.time()
        val_progress = 0
        val_all_preds_phase = []
        val_all_labels_phase = []

        with torch.no_grad():
            for data in val_loader:
                if use_gpu:
                    inputs, labels_phase = data[0].to(device), data[1].to(device)
                    if torch.cuda.memory_reserved() > 0.8 * 8 * 1024 * 1024 * 1024:
                        print(torch.cuda.memory_reserved() // (1024 * 1024 * 1024))
                        return -2
                else:
                    inputs, labels_phase = data[0], data[1]

                labels_phase = labels_phase[(sequence_length - 1)::sequence_length]

                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                outputs_phase = model.forward(inputs)
                outputs_phase = outputs_phase[sequence_length - 1::sequence_length]

                _, preds_phase = torch.max(outputs_phase.data, 1)
                loss_phase = criterion_phase(outputs_phase, labels_phase)

                val_loss_phase += loss_phase.data.item()

                val_corrects_phase += torch.sum(preds_phase == labels_phase.data)
                # TODO

                for i in range(len(preds_phase)):
                    val_all_preds_phase.append(int(preds_phase.data.cpu()[i]))
                for i in range(len(labels_phase)):
                    val_all_labels_phase.append(int(labels_phase.data.cpu()[i]))

                prg_i += 1
                val_progress += 1
                prg_s = prg_i / progressmax * 100
                el_time, el_time_s, _, _ = _functions.run_stats(prg_s, start_time)
                percent = val_progress * val_batch_size / num_val_all * 100
                if epoch == 0:
                    rem_time_s = '0:00:00'
                stats = '\rStatus: {:4.2f}% (validation)  Epoch: {:4.2f}%  Elapsed time: {:8s}  Est time: {:8s} '.format(
                    prg_s, min(100, percent), el_time_s, rem_time_s)
                print(stats, end=' ')

        val_elapsed_time = time.time() - val_start_time
        val_accuracy_phase = float(val_corrects_phase) / float(num_val_we_use)
        val_average_loss_phase = val_loss_phase / num_val_we_use

        val_recall_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average='macro',
                                                zero_division=0)
        val_precision_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average='macro',
                                                      zero_division=0)
        val_jaccard_phase = metrics.jaccard_score(val_all_labels_phase, val_all_preds_phase, average='macro',
                                                  zero_division=0)
        val_precision_each_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average=None,
                                                           zero_division=0)
        val_recall_each_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average=None,
                                                     zero_division=0)
        el_time, el_time_s, rem_time_s, speed = _functions.run_stats(100, start_time)
        epoch_stat = '\rEpoch:  {:3d} /{:3d}   Elapsed time: {}    Estimated remaining time: {} '.format(
            epoch,
            epochs-1,
            el_time_s,
            rem_time_s)
        for x in range(len(stats) - len(epoch_stat) + 1):
            epoch_stat = epoch_stat + ' '
        log(epoch_stat)

        log(' train in          : {:2.0f}m{:2.0f}s\n'
            ' train loss(phase) : {:4.4f}\n'
            ' train accu(phase) : {:.4f}\n'
            ' valid in          : {:2.0f}m{:2.0f}s\n'
            ' valid loss(phase) : {:4.4f}\n'
            ' valid accu(phase) : {:.4f}\n'
            .format(train_elapsed_time // 60,
                    train_elapsed_time % 60,
                    train_average_loss_phase,
                    train_accuracy_phase,
                    val_elapsed_time // 60,
                    val_elapsed_time % 60,
                    val_average_loss_phase,
                    val_accuracy_phase))

        log("val_precision_each_phase   : {}".format(val_precision_each_phase))
        log("val_recall_each_phase      : {}".format(val_recall_each_phase))
        log("val_precision_phase        : {}".format(val_precision_phase))
        log("val_recall_phase           : {}".format(val_recall_phase))
        log("val_jaccard_phase          : {}\n".format(val_jaccard_phase))

        if optimizer_choice == 0:
            if sgd_adjust_lr == 0:
                exp_lr_scheduler.step()
            elif sgd_adjust_lr == 1:
                exp_lr_scheduler.step(val_average_loss_phase)

        if val_accuracy_phase > best_val_accuracy_phase:
            best_val_accuracy_phase = val_accuracy_phase
            correspond_train_acc_phase = train_accuracy_phase
            best_model_wts = copy.deepcopy(model.module.state_dict())
            best_epoch = epoch
            throw = 0
        if val_accuracy_phase == best_val_accuracy_phase:
            if train_accuracy_phase > correspond_train_acc_phase:
                correspond_train_acc_phase = train_accuracy_phase
                best_model_wts = copy.deepcopy(model.module.state_dict())
                best_epoch = epoch
                throw = 0

        save_val_phase = int("{:4.0f}".format(best_val_accuracy_phase * 10000))
        save_train_phase = int("{:4.0f}".format(correspond_train_acc_phase * 10000))
        base_name = "resnetfc_ce" \
                    + "_epoch-" + str(best_epoch) \
                    + "_length-" + str(sequence_length) \
                    + "_batch-" + str(train_batch_size) \
                    + "_train-" + str(save_train_phase) \
                    + "_val-" + str(save_val_phase)

#        if throw >= epochs / 3:
#            log('\nThere was no improvement in accuracy for {} epochs thus the training has ended'.format(
#                int(epochs / 3)))
#            break
    torch.save(best_model_wts, "./best_model/lstm/" + base_name + ".pth")
    log("best_epoch: {}\n".format(str(best_epoch)))
    log(summary(model, verbose=0))
    return 1


if __name__ == "__main__":
    gpu_usg = args.gpu
    sequence_length = args.seq
    train_batch_size = args.train
    val_batch_size = args.val
    optimizer_choice = args.opt
    multi_optim = args.multi
    epochs = args.epo
    workers = args.work
    learning_rate = args.lr
    momentum = args.momentum
    weight_decay = args.weightdecay
    dampening = args.dampening
    use_nesterov = args.nesterov
    sgd_adjust_lr = args.sgdadjust
    sgd_step = args.sgdstep
    sgd_gamma = args.sgdgamma
    num_gpu = torch.cuda.device_count()
    use_gpu = (torch.cuda.is_available() and gpu_usg)
    device = torch.device("cuda:0" if use_gpu else "cpu")

    logfile = '{}\\{}'.format('log', datetime.now().strftime("TransSV_p1_%Y%m%d_%H%M%S.log"))
    _functions.logging.basicConfig(filename=logfile, level=_functions.logging.INFO)

    train_dataset_80, train_num_each_80, val_dataset_80, val_num_each_80, test_dataset_80, test_num_each_80 = _functions.get_data(
        './train_val_paths_labels_21.pkl', sequence_length)
    suc = 0
    while suc != 1:
        torch.cuda.empty_cache()
        suc = train_model((train_dataset_80), (train_num_each_80), (val_dataset_80, test_dataset_80),
                          (val_num_each_80, test_num_each_80))
        ntrbatch = train_batch_size - int(min((train_batch_size) / 2, 8) // 1)
        nvbatch = val_batch_size - int(min((val_batch_size) / 2, 8) // 1)
        if suc == -1:
            log('\nTrain batch number ({}) was too high to fit in memory. Trying lower ({}) number.\n'.format(
                train_batch_size,
                ntrbatch))
            train_batch_size = ntrbatch
        if suc == -2:
            log(
                '\nValidation batch number ({}) was too high to fit in memory. Trying lower ({}) number.\n'.format(
                    val_batch_size, nvbatch))
            val_batch_size = nvbatch

        if train_batch_size < 16:
            print(
                '\nToo large images for training to input into the network, please configure the resize parameters lower')
            break
        if val_batch_size < 1:
            print(
                '\nToo large images for validation to input into the network, please configure the resize parameters lower')
            break
