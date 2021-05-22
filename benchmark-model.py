# Import libraries
import tensorflow as tf
from tensorflow import keras

# Create a basic model to ensure the data was preprocessed correctly
tf.random.set_seed(42)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu', input_shape=[13]),
  tf.keras.layers.Dense(30, activation='relu'),
  tf.keras.layers.Dense(2, activation='sigmoid')
])

model.compile(loss=tf.keras.losses.MAE,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=5)
