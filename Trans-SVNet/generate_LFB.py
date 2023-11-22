# some code adapted from https://github.com/YuemingJin/MTRCNet-CL
# and https://github.com/YuemingJin/TMRNet

import argparse
import pickle
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torchinfo import summary
from torchvision import transforms
from torchvision.transforms import Lambda

import _classes
import _functions
import _models
import _transforms
from _functions import log

parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument('-g', '--gpu', default=True, type=bool, help='gpu use, default True')
parser.add_argument('-s', '--seq', default=1, type=int, help='sequence length, default 10')
parser.add_argument('-t', '--train', default=400, type=int, help='train batch size, default 400')
parser.add_argument('-v', '--val', default=400, type=int, help='valid batch size, default 10')
parser.add_argument('-o', '--opt', default=0, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=25, type=int, help='epochs to train and val, default 25')
parser.add_argument('-w', '--work', default=8, type=int, help='num of workers to use, default 4')
parser.add_argument('-f', '--flip', default=1, type=int, help='0 for not flip, 1 for flip, default 0')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
parser.add_argument('-l', '--lr', default=5e-7, type=float, help='learning rate for optimizer, default 5e-5')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
parser.add_argument('--weightdecay', default=5e-4, type=float, help='weight decay for sgd, default 0')
parser.add_argument('--dampening', default=0, type=float, help='dampening for sgd, default 0')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum, default False')
parser.add_argument('--sgdadjust', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
parser.add_argument('--sgdstep', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
parser.add_argument('--sgdgamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')
parser.add_argument('--LFB_l', default=40, type=int, help='long term feature bank length')
parser.add_argument('--load_LFB', default=False, type=bool, help='whether load exist long term feature bank')

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
    log('test crop type  : {:6d}'.format(crop_type))
    log('whether to flip : {:6d}'.format(use_flip))
    log('learning rate   : {:.4f}'.format(learning_rate))
    log('momentum for sgd: {:.4f}'.format(momentum))
    log('weight decay    : {:.4f}'.format(weight_decay))
    log('dampening       : {:.4f}'.format(dampening))
    log('use nesterov    : {:6d}'.format(use_nesterov))
    log('method for sgd  : {:6d}'.format(sgd_adjust_lr))
    log('step for sgd    : {:6d}'.format(sgd_step))
    log('gamma for sgd   : {:.4f}'.format(sgd_gamma))


g_LFB_train = np.zeros(shape=(0, 2048))
g_LFB_val = np.zeros(shape=(0, 2048))
g_LFB_test = np.zeros(shape=(0, 2048))


def train_model(train_dataset, train_num_each, val_dataset, val_num_each):
    (train_num_each_80), \
        (val_dataset, test_dataset), \
        (val_num_each, test_num_each) = train_num_each, val_dataset, val_num_each

    (train_dataset_80, train_dataset_80_LFB) = train_dataset

    train_useful_start_idx_80 = _functions.get_useful_start_idx(sequence_length, train_num_each_80)
    val_useful_start_idx = _functions.get_useful_start_idx(sequence_length, val_num_each)
    test_useful_start_idx = _functions.get_useful_start_idx(sequence_length, test_num_each)

    train_useful_start_idx_80_LFB = _functions.get_useful_start_idx_LFB(sequence_length, train_num_each_80)
    val_useful_start_idx_LFB = _functions.get_useful_start_idx_LFB(sequence_length, val_num_each)
    test_useful_start_idx_LFB = _functions.get_useful_start_idx_LFB(sequence_length, test_num_each)

    num_train_we_use_80 = len(train_useful_start_idx_80)
    num_val_we_use = len(val_useful_start_idx)
    num_test_we_use = len(test_useful_start_idx)

    num_train_we_use_80_LFB = len(train_useful_start_idx_80_LFB)
    num_val_we_use_LFB = len(val_useful_start_idx_LFB)
    num_test_we_use_LFB = len(test_useful_start_idx_LFB)

    train_we_use_start_idx_80 = train_useful_start_idx_80
    val_we_use_start_idx = val_useful_start_idx
    test_we_use_start_idx = test_useful_start_idx

    train_we_use_start_idx_80_LFB = train_useful_start_idx_80_LFB
    val_we_use_start_idx_LFB = val_useful_start_idx_LFB
    test_we_use_start_idx_LFB = test_useful_start_idx_LFB

    #    np.random.seed(0)
    # np.random.shuffle(train_we_use_start_idx)
    train_idx = []
    for i in range(num_train_we_use_80):
        for j in range(sequence_length):
            train_idx.append(train_we_use_start_idx_80[i] + j)

    val_idx = []
    for i in range(num_val_we_use):
        for j in range(sequence_length):
            val_idx.append(val_we_use_start_idx[i] + j)

    test_idx = []
    for i in range(num_test_we_use):
        for j in range(sequence_length):
            test_idx.append(test_we_use_start_idx[i] + j)

    train_idx_LFB = []
    for i in range(num_train_we_use_80_LFB):
        for j in range(sequence_length):
            train_idx_LFB.append(train_we_use_start_idx_80_LFB[i] + j)

    val_idx_LFB = []
    for i in range(num_val_we_use_LFB):
        for j in range(sequence_length):
            val_idx_LFB.append(val_we_use_start_idx_LFB[i] + j)

    test_idx_LFB = []
    for i in range(num_test_we_use_LFB):
        for j in range(sequence_length):
            test_idx_LFB.append(test_we_use_start_idx_LFB[i] + j)

    dict_index, dict_value = zip(*list(enumerate(train_we_use_start_idx_80_LFB)))
    dict_train_start_idx_LFB = dict(zip(dict_value, dict_index))

    dict_index, dict_value = zip(*list(enumerate(val_we_use_start_idx_LFB)))
    dict_val_start_idx_LFB = dict(zip(dict_value, dict_index))

    num_train_all = len(train_idx)
    num_val_all = len(val_idx)
    num_test_all = len(test_idx)

    log('num train start idx 80: {:6d}'.format(len(train_useful_start_idx_80)))
    log('num of all train use: {:6d}'.format(num_train_all))
    log('num of all valid use: {:6d}'.format(num_val_all))
    log('num of all test use: {:6d}'.format(num_test_all))
    log('num of all train LFB use: {:6d}'.format(len(train_idx_LFB)))
    log('num of all valid LFB use: {:6d}'.format(len(val_idx_LFB)))

    global g_LFB_train
    global g_LFB_val
    global g_LFB_test
    print("loading features!>.........")

    if not load_exist_LFB:

        train_feature_loader = DataLoader(
            train_dataset_80_LFB,
            batch_size=val_batch_size,
            sampler= _transforms.SeqSampler(train_dataset_80_LFB, train_idx_LFB),
            num_workers=workers,
            pin_memory=False
        )
        val_feature_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=_transforms.SeqSampler(val_dataset, val_idx_LFB),
            num_workers=workers,
            pin_memory=False
        )

        test_feature_loader = DataLoader(
            test_dataset,
            batch_size=val_batch_size,
            sampler=_transforms.SeqSampler(test_dataset, test_idx_LFB),
            num_workers=workers,
            pin_memory=False
        )

        model_LFB = _models.resnet_lstm_LFB()
        modelpth = input('\nPlease enter the ResNet model name (*.pth):\n').split('.')[0]
        model_LFB.load_state_dict(torch.load("./best_model/lstm/" + modelpth + ".pth"), strict=False)

        def get_parameter_number(net):
            total_num = sum(p.numel() for p in net.parameters())
            trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
            return trainable_num

        log('')
        log('START training - {}\n'.format(time.strftime("%Y-%m-%d %H:%M:%S")))
        total_papa_num = 0
        total_papa_num += get_parameter_number(model_LFB)

        if use_gpu:
            model_LFB = DataParallel(model_LFB)
            model_LFB.to(device)

        for params in model_LFB.parameters():
            params.requires_grad = False

        from torchinfo import summary
        log(summary(model_LFB, verbose=0))
        log('\n\n')
        input('waithere')

        model_LFB.eval()

        with torch.no_grad():
            # '''
            i = 0
            for data in train_feature_loader:
                i += 1
                if use_gpu:
                    inputs, labels_phase = data[0].to(device), data[1].to(device)
                else:
                    inputs, labels_phase = data[0], data[1]

                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                outputs_feature = model_LFB.forward(inputs).data.cpu().numpy()

                g_LFB_train = np.concatenate((g_LFB_train, outputs_feature), axis=0)

                print("\rStatus: {:4.2f}%  train feature length: {} ".format(i/len(train_feature_loader)*100,len(g_LFB_train)), end='')

            for data in val_feature_loader:
                if use_gpu:
                    inputs, labels_phase = data[0].to(device), data[1].to(device)
                else:
                    inputs, labels_phase = data[0], data[1]

                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                outputs_feature = model_LFB.forward(inputs).data.cpu().numpy()

                g_LFB_val = np.concatenate((g_LFB_val, outputs_feature), axis=0)

                print("\rStatus: {:4.2f}%  val feature length: {} ".format(i/len(val_feature_loader)*100,len(g_LFB_val)), end='')
            # '''
            for data in test_feature_loader:
                if use_gpu:
                    inputs, labels_phase = data[0].to(device), data[1].to(device)
                else:
                    inputs, labels_phase = data[0], data[1]

                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                outputs_feature = model_LFB.forward(inputs).data.cpu().numpy()

                g_LFB_test = np.concatenate((g_LFB_test, outputs_feature), axis=0)

                print("\rStatus: {:4.2f}%  test feature length: {} ".format(i/len(test_feature_loader)*100,len(g_LFB_test)), end='')

        print("\nfinish!")
        g_LFB_train = np.array(g_LFB_train)
        g_LFB_val = np.array(g_LFB_val)
        g_LFB_test = np.array(g_LFB_test)
        # '''
        with open("./LFB/g_LFB50_train0.pkl", 'wb') as f:
            pickle.dump(g_LFB_train, f)

        with open("./LFB/g_LFB50_val0.pkl", 'wb') as f:
            pickle.dump(g_LFB_val, f)
        # '''
        with open("./LFB/g_LFB50_test0.pkl", 'wb') as f:
            pickle.dump(g_LFB_test, f)
        log(summary(model_LFB, verbose=0))


if __name__ == "__main__":
    gpu_usg = args.gpu
    sequence_length = args.seq
    train_batch_size = args.train
    val_batch_size = args.val
    optimizer_choice = args.opt
    multi_optim = args.multi
    epochs = args.epo
    workers = args.work
    use_flip = args.flip
    crop_type = args.crop
    learning_rate = args.lr
    momentum = args.momentum
    weight_decay = args.weightdecay
    dampening = args.dampening
    use_nesterov = args.nesterov

    LFB_length = args.LFB_l
    load_exist_LFB = args.load_LFB

    sgd_adjust_lr = args.sgdadjust
    sgd_step = args.sgdstep
    sgd_gamma = args.sgdgamma

    num_gpu = torch.cuda.device_count()
    use_gpu = (torch.cuda.is_available() and gpu_usg)
    device = torch.device("cuda:0" if use_gpu else "cpu")

    logfile = '{}\\{}'.format('log', datetime.now().strftime("TransSV_p2_%Y%m%d_%H%M%S.log"))
    _functions.logging.basicConfig(filename=logfile, level=_functions.logging.INFO)

    print_params()
    train_dataset_80, train_num_each_80, \
        val_dataset, val_num_each, test_dataset, test_num_each = _functions.get_data_LFB('./train_val_paths_labels_21.pkl', sequence_length)

    train_model((train_dataset_80),
                (train_num_each_80),
                (val_dataset, test_dataset),
                (val_num_each, test_num_each))
