import logging
import pickle
import time

import numpy as np
from PIL import Image
from torchvision import transforms

import _classes
import _transforms


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


def get_useful_start_idx_LFB(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx


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

def get_long_feature(start_index, lfb, LFB_length):
    long_feature = []
    long_feature_each = []
    # 上一个存在feature的index
    for k in range(LFB_length):
        LFB_index = (start_index + k)
        LFB_index = int(LFB_index)
        long_feature_each.append(lfb[LFB_index])
    long_feature.append(long_feature_each)
    return long_feature

def get_data(data_path, sequence_length):
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

    print('train_paths  : {:6d}'.format(len(train_paths)))
    print('train_labels : {:6d}'.format(len(train_labels)))
    print('valid_paths  : {:6d}'.format(len(val_paths)))
    print('valid_labels : {:6d}'.format(len(val_labels)))
    print('test_paths  : {:6d}'.format(len(test_paths)))
    print('test_labels : {:6d}'.format(len(test_labels)))

    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    train_transforms = None
    test_transforms = None
    normalize_m = ([0.41757566, 0.26098573, 0.25888634])
    normalize_s = ([0.21938758, 0.1983, 0.19342837])
    resize = (240, 240)
    crop = 224

    train_transforms = transforms.Compose([
        transforms.Resize(resize),
        _transforms.RandomCrop(crop, sequence_length),
        _transforms.ColorJitter(sequence_length, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        _transforms.RandomHorizontalFlip(sequence_length),
        _transforms.RandomRotation(5, sequence_length),
        transforms.ToTensor(),
        transforms.Normalize(normalize_m, normalize_s)])

    test_transforms = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(normalize_m, normalize_s)])

    train_dataset_80 = _classes.CholecDataset(train_paths, train_labels, train_transforms)
    val_dataset_80 = _classes.CholecDataset(val_paths, val_labels, test_transforms)
    test_dataset_80 = _classes.CholecDataset(test_paths, test_labels, test_transforms)

    return train_dataset_80, train_num_each, val_dataset_80, val_num_each, test_dataset_80, test_num_each


def get_data2(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    train_paths_80 = train_test_paths_labels[0]
    val_paths_80 = train_test_paths_labels[1]
    test_paths_80 = train_test_paths_labels[2]

    train_labels_80 = train_test_paths_labels[3]
    val_labels_80 = train_test_paths_labels[4]
    test_labels_80 = train_test_paths_labels[5]

    train_num_each_80 = train_test_paths_labels[6]
    val_num_each_80 = train_test_paths_labels[7]
    test_num_each_80 = train_test_paths_labels[8]

    print('train_paths_80  : {:6d}'.format(len(train_paths_80)))
    print('train_labels_80 : {:6d}'.format(len(train_labels_80)))
    print('valid_paths_80  : {:6d}'.format(len(val_paths_80)))
    print('valid_labels_80 : {:6d}'.format(len(val_labels_80)))
    print('test_paths_80  : {:6d}'.format(len(test_paths_80)))
    print('test_labels_80 : {:6d}'.format(len(test_labels_80)))

    train_labels_80 = np.asarray(train_labels_80, dtype=np.int64)
    val_labels_80 = np.asarray(val_labels_80, dtype=np.int64)
    test_labels_80 = np.asarray(test_labels_80, dtype=np.int64)

    train_start_vidx = []
    count = 0
    for i in range(len(train_num_each_80)):
        train_start_vidx.append(count)
        count += train_num_each_80[i]

    val_start_vidx = []
    count = 0
    for i in range(len(val_num_each_80)):
        val_start_vidx.append(count)
        count += val_num_each_80[i]

    test_start_vidx = []
    count = 0
    for i in range(len(test_num_each_80)):
        test_start_vidx.append(count)
        count += test_num_each_80[i]

    return train_labels_80, train_num_each_80, train_start_vidx, val_labels_80, val_num_each_80, val_start_vidx, \
        test_labels_80, test_num_each_80, test_start_vidx


def get_data_LFB(data_path, sequence_length):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    train_paths = train_test_paths_labels[0]
    val_paths = train_test_paths_labels[1]
    test_paths = train_test_paths_labels[2]

    train_labels = train_test_paths_labels[3]
    val_labels = train_test_paths_labels[4]
    test_labels = train_test_paths_labels[5]

    train_num_each_80 = train_test_paths_labels[6]
    val_num_each_80 = train_test_paths_labels[7]
    test_num_each_80 = train_test_paths_labels[8]

    print('train_paths  : {:6d}'.format(len(train_paths)))
    print('train_labels : {:6d}'.format(len(train_labels)))
    print('valid_paths  : {:6d}'.format(len(val_paths)))
    print('valid_labels : {:6d}'.format(len(val_labels)))
    print('test_paths  : {:6d}'.format(len(test_paths)))
    print('test_labels : {:6d}'.format(len(test_labels)))

    train_labels_80 = np.asarray(train_labels, dtype=np.int64)
    val_labels_80 = np.asarray(val_labels, dtype=np.int64)
    test_labels_80 = np.asarray(test_labels, dtype=np.int64)

    train_transforms = None
    test_transforms = None

    train_transforms = transforms.Compose([
        transforms.Resize((250, 250)),
        _transforms.RandomCrop(224, sequence_length),
        _transforms.ColorJitter(sequence_length, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        _transforms.RandomHorizontalFlip(sequence_length),
        _transforms.RandomRotation(5, sequence_length),
        transforms.ToTensor(),
        transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
    ])

    train_dataset_80 = _classes.CholecDataset(train_paths, train_labels_80, train_transforms)
    train_dataset_80_LFB = _classes.CholecDataset(train_paths, train_labels_80, test_transforms)
    val_dataset_80 = _classes.CholecDataset(val_paths, val_labels_80, test_transforms)
    test_dataset_80 = _classes.CholecDataset(test_paths, test_labels_80, test_transforms)

    return (train_dataset_80, train_dataset_80_LFB), train_num_each_80, \
        val_dataset_80, val_num_each_80, test_dataset_80, test_num_each_80
