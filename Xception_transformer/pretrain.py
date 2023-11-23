rt# %%
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds


# %%
import pickle

# %%
with open('test_X.pkl', 'rb') as f:
    test_X = pickle.load(f)
with open('test_y.pkl', 'rb') as f:
    test_y = pickle.load(f)

# %%

builder_train = tfds.folder_dataset.ImageFolder('pretrain/videos/')
raw_train = builder_train.as_dataset(split="train", shuffle_files=True)
builder_test = tfds.folder_dataset.ImageFolder('pretrain/videos/')
raw_test = builder_test.as_dataset(split="test", shuffle_files=True)
builder_val = tfds.folder_dataset.ImageFolder('pretrain/videos/')
raw_val = builder_val.as_dataset(split="val", shuffle_files=True)

# %%
IMG_SIZE = 160

def format_example(pair):
  image, label = pair['image'], pair['label']
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, tf.one_hot(label, 7)

# %%
train = raw_train.map(format_example)
validation = raw_val.map(format_example)
test = raw_test.map(format_example)

# %%
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE =1000

# %%
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# %%
for image_batch, label_batch in train_batches.take(1):
   pass

image_batch.shape

# %%
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = tf.keras.applications.Xception(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


# %%
feature_batch = base_model(image_batch)
print(feature_batch.shape)

# %%
base_model.trainable = False

# %%
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)

# %%
prediction_layer = tf.keras.layers.Dense(7)
prediction_batch = prediction_layer(feature_batch_average)

# %%
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

# %%
base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits = True),
              metrics=['accuracy'])

# %%
initial_epochs = 10
validation_steps = 10
loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

# %%
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

# %%

history = model.fit(train_batches,
                        epochs=initial_epochs, 
                        validation_data=validation_batches)


# %%
base_model.trainable = True

# %%
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

# %%
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate/10),
              metrics=['accuracy'])

# %%
model.summary()

# %%
fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_batches,
                         epochs=total_epochs,
                         initial_epoch =  history.epoch[-1],
                         validation_data=validation_batches)

# %%
model.layers[0].save_weights('xception.h5')

