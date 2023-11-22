import logging
import pickle
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from pprint import pprint
import _classes as classes


def run_stats(prg_s, start_time):
    el_time = time.time() - start_time
    speed = el_time / prg_s
    rem_time = (100 - prg_s) * speed
    el_time_s = timesplit(el_time)
    rem_time_s = timesplit(rem_time)
    return el_time, el_time_s, rem_time_s, speed


def log(text):
    logging.info(text)
    print(text)


def timesplit(x):
    mon, sec = divmod(x, 60)
    hr, mon = divmod(mon, 60)
    return "%d:%02d:%02d" % (hr, mon, sec)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def get_useful_start_idx(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx


def get_data(data_path, params):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    train_paths = train_test_paths_labels[0]
    val_paths = train_test_paths_labels[1]
    test_paths = train_test_paths_labels[2]
    train_labels = train_test_paths_labels[3]
    val_labels = train_test_paths_labels[4]
    test_labels = train_test_paths_labels[5]
    train_num_each = train_test_paths_labels[6]
    val_num_each = train_test_paths_labels[7]
    test_num_each = train_test_paths_labels[8]

    log('train_paths  : {:6d}'.format(len(train_paths)))
    log('train_labels : {:6d}'.format(len(train_labels)))
    log('valid_paths  : {:6d}'.format(len(val_paths)))
    log('valid_labels : {:6d}'.format(len(val_labels)))
    log('test_paths   : {:6d}'.format(len(test_paths)))
    log('test_labels  : {:6d}'.format(len(test_labels)))

    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    train_transform = transforms.Compose([
        classes.ResizeImg(params.get_train_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
    ])
    valid_transform = transforms.Compose([
        classes.ResizeImg(params.get_test_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
    ])
    test_transform = transforms.Compose([
        classes.ResizeImg(params.get_test_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
    ])

    train_dataset = classes.CholecDataset(train_paths, train_labels, train_transform)
    val_dataset = classes.CholecDataset(val_paths, val_labels, valid_transform)
    test_dataset = classes.CholecDataset(test_paths, test_labels, test_transform)

    return train_dataset, train_num_each, val_dataset, val_num_each, test_dataset, test_num_each


def print_params(params):
    pprint(vars(params))
    log('training videos     : {:6d}'.format(params.get_train_set))
    log('validation videos   : {:6d}'.format(params.get_valid_set))
    log('test videos         : {:6d}'.format(params.get_test_set))
    log('og img resize factor: {:.2f}'.format(params.get_resize))
    log('train resize factor : {:.2f}'.format(params.get_resize))
    log('test factor         : {:.2f}'.format(params.get_resize))
    log('num of epochs       : {:6d}'.format(params.get_epochs))
    log('sequence length     : {:6d}'.format(params.get_sequence))
    log('train batch size    : {:6d}'.format(params.get_train_batch))
    log('valid batch size    : {:6d}'.format(params.get_valid_batch))
    log('test batch size     : {:6d}'.format(params.get_test_batch))
    log('optimizer choice    : {:s}'.format(params.get_optimizer))
    log('num of workers      : {:6d}'.format(params.get_workers))
    log('learning rate       : {:.4f}'.format(params.get_lr))
    log('weight decay        : {:.4f}'.format(params.get_weight_decay))
    log('dampening           : {:.4f}'.format(params.get_dampening))
    log('torch version       : {:6s}'.format(torch.__version__))
    log('number of gpu       : {:6d}'.format(params.get_num_gpu))
  #  for i in range(params.get_gpu+1):
     #   log('gpu                 : {}'.format(torch.cuda.get_device_name(cuda:0)))


def load_params():
    params = classes.Params()
    return params


def data_loader(dataset, idx, batch, workers):
    loader = DataLoader(dataset,
                              batch_size=batch,
                              sampler=idx,
                              num_workers=workers,
                              pin_memory=True
                        )
    return loader
