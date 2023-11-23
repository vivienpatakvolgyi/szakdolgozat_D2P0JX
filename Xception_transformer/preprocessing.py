# %%
import os
import time
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings("ignore")
import cv2
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd


# %%
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# %%
import os

# %%
train_X_path = []
val_X_path = []
test_X_path =[]

train_y_names = []
val_y_names = []
test_y_names = []

for dirname, _, filenames in tqdm(os.walk('videos')): 
    for filename in sorted(filenames, key = natural_keys):
        
        path = os.path.join(dirname, filename)

        num = str(((path.split(os.path.sep)[-1]).split('video')[-1]).split('-')[0])
        frame = int((path.split(os.path.sep)[-1]).split('-')[-1].split('.')[-2])

        f=open(f'labels\label_{num}.txt')
        lines = f.readlines()
        

        if 'train' in dirname:
            train_X_path.append(path)
            train_y_names.append((str(lines).split(','))[frame].split('\\')[1][-1])

        elif 'val' in dirname:
            val_X_path.append(path)
            val_y_names.append((str(lines).split(','))[frame].split('\\')[1][-1])

        elif 'test' in dirname:
            test_X_path.append(path)
            test_y_names.append((str(lines).split(','))[frame].split('\\')[1][-1])



# %%
def img_prep(features, output, dims):

    img_data = []
    labels = []

    for enum, imagePath in tqdm(enumerate(features)):
    
        try:
            counter = 0
            img=cv2.imread(imagePath)
            img=cv2.resize(img, (dims[1], dims[0]))
            
        except Exception as e:
        
            counter = 1
    
        if counter==0:
            
            label = output[enum]
            labels.append(label)
        
            img_data.append(img)
            
    return img_data, labels

# %%

IMAGE_DIMS = (240, 240, 3)


# %%
val_X, val_y = img_prep(val_X_path, val_y_names, IMAGE_DIMS)

# %%
test_X, test_y = img_prep(test_X_path, test_y_names, IMAGE_DIMS)

# %%
train_X, train_y = img_prep(train_X_path, train_y_names, IMAGE_DIMS)

# %%
import pickle 
with open('train_X.pkl', 'wb') as f:
    pickle.dump(train_X, f)
with open('test_X.pkl', 'wb') as f:
    pickle.dump(test_X, f)
with open('val_X.pkl', 'wb') as f:
    pickle.dump(val_X, f)


# %%
with open('train_y.pkl', 'wb') as f:
    pickle.dump(train_y, f)
with open('test_y.pkl', 'wb') as f:
    pickle.dump(test_y, f)
with open('val_y.pkl', 'wb') as f:
    pickle.dump(val_y, f)


