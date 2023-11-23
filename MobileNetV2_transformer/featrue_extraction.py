# %%
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
import pickle

# %%
def img_feature_extraction(x_values, pre_model):

    data = []
    for image in tqdm(x_values):
        im_toarray = tf.keras.preprocessing.image.img_to_array(image)
        
        im_toarray = np.expand_dims(image, axis=0)
        im_toarray = tf.keras.applications.mobilenet_v2.preprocess_input(im_toarray)
        
        data.append(im_toarray)
        
    data_stack = np.vstack(data) 
    
    features = pre_model.predict(data_stack, batch_size=52)
    
    return data_stack, features

# %%
IMAGE_DIMS = (240, 240, 3)

# %%

mnet_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_DIMS, include_top=False, weights='imagenet')
    
for layer in mnet_model.layers:
    layer.trainable = False

# %%
mnet_model.load_weights('mobilenet.h5')

# %%
with open('train_X.pkl', 'rb') as f:
    train_X = pickle.load(f)
with open('train_y.pkl', 'rb') as f:
    train_y = pickle.load(f)

# %%
with open('test_X.pkl', 'rb') as f:
    test_X = pickle.load(f)
with open('test_y.pkl', 'rb') as f:
    test_y = pickle.load(f)

# %%
with open('val_X.pkl', 'rb') as f:
    val_X = pickle.load(f)
with open('val_y.pkl', 'rb') as f:
    val_y = pickle.load(f)

# %%
len(test_y)

# %%
frames_test = [3546, 3413]
frames_val = [4832, 4326]
frames_train = [6388, 3620, 3000, 2938, 3220, 3908, 1645, 4692, 5736, 5064, 4720, 2916, 2597, 4739, 3653, 3612, 4678]

# %%
def extract(frames, model, feature_data, label_data, name):
    labels = []

    start = 0
    for i in range(len(frames)):
        if i > -1:
            if start == 0:
                p, f = img_feature_extraction(feature_data[ :frames[i]], model)
                labels.append(label_data[ :frames[i]])

                
            else:
                p, f = img_feature_extraction(feature_data[start:start+frames[i]], model)
                labels.append(label_data[start:start+frames[i]])



            with open(f'{name}_extract{i}.pkl', 'wb') as file:  # open a text file
                pickle.dump(f, file) 
        
        start += frames[i]

    with open(f'{name}_extract_y.pkl', 'wb') as file:  # open a text file
        pickle.dump(labels, file)


# %%

extract(frames_test, mnet_model, test_X, test_y, 'test')


# %%
extract(frames_val, mnet_model, val_X, 'val')

# %%
extract(frames_train, mnet_model, train_X, 'train')


