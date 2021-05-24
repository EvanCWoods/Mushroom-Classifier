# Import libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import layers
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras import models
from keras.models import Sequential
import random
import pandas as pd
import numpy as np

# Import the data
mushroom_data = pd.read_csv('mushrooms.csv')

def numerize_data():
  mushroom_data['Mushroom'] = pd.Categorical(mushroom_data['Mushroom'])
  mushroom_data['Mushroom'] = mushroom_data.Mushroom.cat.codes

  mushroom_data['capShape'] = pd.Categorical(mushroom_data['capShape'])
  mushroom_data['capShape'] = mushroom_data.capShape.cat.codes

  mushroom_data['capSurface'] = pd.Categorical(mushroom_data['capSurface'])
  mushroom_data['capSurface'] = mushroom_data.capSurface.cat.codes

  mushroom_data['capColor'] = pd.Categorical(mushroom_data['capColor'])
  mushroom_data['capColor'] = mushroom_data.capColor.cat.codes

  mushroom_data['bruises'] = pd.Categorical(mushroom_data['bruises'])
  mushroom_data['bruises'] = mushroom_data.bruises.cat.codes

  mushroom_data['odor'] = pd.Categorical(mushroom_data['odor'])
  mushroom_data['odor'] = mushroom_data.odor.cat.codes

  mushroom_data['gillAttachment'] = pd.Categorical(mushroom_data['gillAttachment'])
  mushroom_data['gillAttachment'] = mushroom_data.gillAttachment.cat.codes

  mushroom_data['gillSpacing'] = pd.Categorical(mushroom_data['gillSpacing'])
  mushroom_data['gillSpacing'] = mushroom_data.gillSpacing.cat.codes

  mushroom_data['gillSize'] = pd.Categorical(mushroom_data['gillSize'])
  mushroom_data['gillSize'] = mushroom_data.gillSize.cat.codes

  mushroom_data['gillColor'] = pd.Categorical(mushroom_data['gillColor'])
  mushroom_data['gillColor'] = mushroom_data.gillColor.cat.codes

  mushroom_data['stalkShape'] = pd.Categorical(mushroom_data['stalkShape'])
  mushroom_data['stalkShape'] = mushroom_data.stalkShape.cat.codes

  mushroom_data['stalkRoot'] = pd.Categorical(mushroom_data['stalkRoot'])
  mushroom_data['stalkRoot'] = mushroom_data.stalkRoot.cat.codes

  mushroom_data['stalkSurfaceAboveRing'] = pd.Categorical(mushroom_data['stalkSurfaceAboveRing'])
  mushroom_data['stalkSurfaceAboveRing'] = mushroom_data.stalkSurfaceAboveRing.cat.codes

  mushroom_data['stalkSurfaceBelowRing'] = pd.Categorical(mushroom_data['stalkSurfaceBelowRing'])
  mushroom_data['stalkSurfaceBelowRing'] = mushroom_data.stalkSurfaceBelowRing.cat.codes

  mushroom_data['stalkColorAboveRing'] = pd.Categorical(mushroom_data['stalkColorAboveRing'])
  mushroom_data['stalkColorAboveRing'] = mushroom_data.stalkColorAboveRing.cat.codes

  mushroom_data['stalkColorBelowRing'] = pd.Categorical(mushroom_data['stalkColorBelowRing'])
  mushroom_data['stalkColorBelowRing'] = mushroom_data.stalkColorBelowRing.cat.codes

  mushroom_data['veilType'] = pd.Categorical(mushroom_data['veilType'])
  mushroom_data['veilType'] = mushroom_data.veilType.cat.codes

  mushroom_data['veilColor'] = pd.Categorical(mushroom_data['veilColor'])
  mushroom_data['veilColor'] = mushroom_data.veilColor.cat.codes

  mushroom_data['ringNumber'] = pd.Categorical(mushroom_data['ringNumber'])
  mushroom_data['ringNumber'] = mushroom_data.ringNumber.cat.codes

  mushroom_data['ringType'] = pd.Categorical(mushroom_data['ringType'])
  mushroom_data['ringType'] = mushroom_data.ringType.cat.codes

  mushroom_data['sporePrintColor'] = pd.Categorical(mushroom_data['sporePrintColor'])
  mushroom_data['sporePrintColor'] = mushroom_data.sporePrintColor.cat.codes

  mushroom_data['population'] = pd.Categorical(mushroom_data['population'])
  mushroom_data['population'] = mushroom_data.population.cat.codes

  mushroom_data['habitat'] = pd.Categorical(mushroom_data['habitat'])
  mushroom_data['habitat'] = mushroom_data.habitat.cat.codes
 
numerize_data()

# Create the X dataset
X = mushroom_data.drop(['Mushroom', 'gillAttachment', 'gillSpacing', 'stalkSurfaceAboveRing', 'stalkSurfaceBelowRing', 'stalkColorAboveRing', 'stalkColorBelowRing', 'veilType', 'veilColor', 'ringNumber'], axis=1)

# Create the y dataset
y = mushroom_data['Mushroom']

# Set the size of the training and testing sets
train_size = round(0.8 * mushroom_data.shape[0])
test_size = round(0.2 * mushroom_data.shape[0])
print(train_size, test_size)

# Create the training and testing datasets
train_data = X[:train_size]
train_labels = y[:train_size]
test_data = X[train_size:]
test_labels = y[train_size:]

# Print the [0] shape of each dataset to verify the correct size
print(train_data.shape[0])
print(train_labels.shape[0])
print(test_data.shape[0])
print(test_labels.shape[0])

tf.random.set_seed(42)

model_1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[13]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(48, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_1.compile(loss=tf.keras.losses.MSE,
optimizer=tf.keras.optimizers.Adam(),
metrics=['accuracy'])

history_1 = model_1.fit(train_data, train_labels, epochs=25, validation_data=(test_data, test_labels))


def plot_history(history):
    plt.plot(history.history['accuracy'], c='b')
    plt.plot(history.history['val_accuracy'], c='g')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

plot_history(history=history_1)
