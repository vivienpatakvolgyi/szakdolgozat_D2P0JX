# %%
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf


# %%
IMAGE_DIMS = (240, 240, 3)

# %%
import pickle

# %%
with open('./train_extract_y.pkl', 'rb') as f:
    train_y = pickle.load(f)

with open('./test_extract_y.pkl', 'rb') as f:
    test_y = pickle.load(f)

# %%
def add(group, amount):
    for i in range(amount):
        with open(f'./{group}_extract{i}.pkl', 'rb') as f:
            X = pickle.load(f)
        X = X.reshape(X.shape[0], 64, X.shape[-1])

        yield X


# %%
test_X = list(add('test', 2))
val_X = list(add('val', 2))

# %%
frames_test = [3546, 3413]
frames_val = [4832, 4326]
frames_train = [6388, 3620, 3000, 2938, 3220, 3908, 1645, 4692, 5736, 5064, 4720, 2916, 2597, 4739, 3653, 3612, 4678]

# %%
#from tensorflow_docs.vis import embed
from tensorflow import keras
#from imutils import paths
from keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os

# %%
MAX_SEQ_LENGTH = 64
NUM_FEATURES = 2048
IMG_SIZE = 240

EPOCHS = 1

# %%
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask

# %%
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

# %%
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

# %%
 
def get_compiled_model():
    sequence_length = MAX_SEQ_LENGTH
    embed_dim = NUM_FEATURES
    dense_dim = 8
    num_heads = 1
    classes = 7

    inputs = keras.Input(shape=(None, None))
    x = PositionalEmbedding(
        sequence_length, embed_dim, name="frame_position_embedding"
    )(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model

 
def run_experiment():
    checkpoint = keras.callbacks.ModelCheckpoint(
        "video_classifier_xception.h5", save_weights_only=True, save_best_only=True, verbose=1
    )

    model = get_compiled_model()
    for i in range(0,17,1):
        with open(f'./train_extract{i}.pkl', 'rb') as f:
            X = pickle.load(f)
        X = X.reshape(X.shape[0], 64, X.shape[-1])
        train_y_a = train_y[i]
        N=1000
        train_X_subs = [X[n:n+N] for n in range(0, len(X), N-100)]
        train_y_subs = [train_y_a[n:n+N] for n in range(0, len(train_y_a), N-100)]
        for k in range(len(train_X_subs)):
            for j in range(2):
                test_X_a = test_X[j]
                test_y_a = test_y[j]
                N=1000
                test_X_subs = [test_X_a[n:n+N] for n in range(0, len(test_X_a), N-100)]
                test_y_subs = [test_y_a[n:n+N] for n in range(0, len(test_y_a), N-100)]
                for l in range(len(test_X_subs)):
                    model.fit(
                        np.array(train_X_subs[k]),
                        np.array(train_y_subs[k]),
                        epochs=1,
                        callbacks=[checkpoint],
                        validation_data = (test_X_subs[l], test_y_subs[l])
                    )
    return model

# %%
trained_model = run_experiment()

# %%
ready_model = get_compiled_model()
ready_model.load_weights('video_classifier_xception.h5')

# %%
y_pred1 = ready_model.predict(val_X[0])
y_pred2 = ready_model.predict(val_X[1])

# %%
with open('Xcp_pred_1.txt', 'w') as f:
    for i in range(y_pred1.shape[0]):
        f.writelines(str(f"{i}\t{np.argmax(y_pred1[i])}\n"))

# %%
with open('Xcp_pred_2.txt', 'w') as f:
    for i in range(y_pred2.shape[0]):
        f.writelines(str(f"{i}\t{np.argmax(y_pred2[i])}\n")) 



