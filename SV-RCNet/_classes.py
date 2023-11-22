import os
import random

import torch
from PIL import Image, ImageOps
from dotenv import load_dotenv
from torch.utils.data import Dataset
import numpy as np


class Params:
    def __init__(self) -> None:
        if os.path.isfile(".env"):
            load_dotenv()
        else:
            print('Cannot find environment file\n')
        self._gpu: int = os.getenv('GPU')
        self._train_set: int = os.getenv('TRAIN_SET')
        self._valid_set: int = os.getenv('VALID_SET')
        self._test_set: int = os.getenv('TEST_SET')
        self._sequence: int = os.getenv('SEQUENCE')
        self._train_batch: int = os.getenv('TRAIN_BATCH')
        self._valid_batch: int = os.getenv('VALID_BATCH')
        self._test_batch: int = os.getenv('TEST_BATCH')
        self._epochs: int = os.getenv('EPOCHS')
        self._workers: int = os.getenv('WORKERS')
        self._lr: float = os.getenv('LR')
        self._lr_decay: float = os.getenv('LR_DECAY')
        #self._momentum: float = os.getenv('MOMENTUM')
        self._weight_decay: float = os.getenv('WEIGHTDECAY')
        self._dampening: float = os.getenv('DAMPENING')
        self._classes: int = os.getenv('NUM_CLASSES')
        self._data_dir: str = os.getenv('DATA_DIR')
        self._log_dir: str = os.getenv('LOG_DIR')
        self._num_gpu: int = torch.cuda.device_count()
        self._use_gpu: bool = torch.cuda.is_available()
        self._optimizer: str = os.getenv('OPTIMIZER')
        self._resize: float = os.getenv('IMG_RESIZE')
        self._train_resize: float = os.getenv('TRAIN_RESIZE')
        self._test_resize: float = os.getenv('TEST_RESIZE')

    @property
    def get_gpu(self):
        return int(self._gpu)

    @property
    def get_train_set(self):
        return int(self._train_set)

    @property
    def get_valid_set(self):
        return int(self._valid_set)

    @property
    def get_test_set(self):
        return int(self._test_set)

    @property
    def get_sequence(self):
        return int(self._sequence)

    @property
    def get_train_batch(self):
        return int(self._train_batch)

    @property
    def get_valid_batch(self):
        return int(self._valid_batch)

    @property
    def get_test_batch(self):
        return int(self._test_batch)

    @property
    def get_epochs(self):
        return int(self._epochs)

    @property
    def get_workers(self):
        return int(self._workers)

    @property
    def get_lr(self):
        return float(self._lr)

    @property
    def get_lr_decay(self):
        return float(self._lr_decay)

    #@property
   # def get_momentum(self):
     #   return float(self._momentum)

    @property
    def get_weight_decay(self):
        return float(self._weight_decay)

    @property
    def get_dampening(self):
        return float(self._dampening)

    @property
    def get_classes(self):
        return int(self._classes)

    @property
    def get_data_dir(self):
        return str(self._data_dir)

    @property
    def get_log_dir(self):
        return str(self._log_dir)

    @property
    def get_num_gpu(self):
        return int(self._num_gpu)

    @property
    def get_use_gpu(self):
        return bool(self._use_gpu)

    @property
    def get_optimizer(self):
        return str(self._optimizer)

    @property
    def get_resize(self):
        return float(self._resize)

    @property
    def get_train_resize(self):
        return float(self._train_resize)

    @property
    def get_test_resize(self):
        return float(self._test_resize)


class ResizeImg(object):
    def __init__(self, factor, padding=0):
        self.count = 0
        self.factor = factor

    def __call__(self, img):
        self.count += 1
        img = ImageOps.scale(img, self.factor)
        return img


class CholecDataset(Dataset):
    def pil_loader(path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def __init__(self, file_paths, file_labels, transform=None,
                 loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels = file_labels[:, -1]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels = self.file_labels[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels

    def __len__(self):
        return len(self.file_paths)
