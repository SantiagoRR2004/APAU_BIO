from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_valid, y_valid) = mnist.load_data()

import matplotlib.pyplot as plt
import numpy as np

n_to_show = 10
input_idx = np.random.choice(range(len(x_train)), n_to_show)
input_images = x_train[input_idx]
input_labels = y_train[input_idx]

fig = plt.figure(figsize=(15, 2))
fig.subplots_adjust(hspace=0.1, wspace=0.4)
for i in range(n_to_show):
    ax = fig.add_subplot(1, n_to_show, i + 1)
    ax.axis("off")
    ax.set_title(input_labels[i])
    fig.suptitle("MNIST digits")
    ax.imshow(input_images[i], cmap="binary")
# plt.show()

x_train = x_train / 255
x_valid = x_valid / 255
print(x_train.dtype)
print(x_train.min(), x_train.max())
x_train = x_train.reshape(60000, 784)
x_valid = x_valid.reshape(10000, 784)
print(x_train.shape)

import tensorflow.keras as keras

num_categories = 10
y_train = keras.utils.to_categorical(y_train, num_categories)
y_valid = keras.utils.to_categorical(y_valid, num_categories)
y_train[0:10]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(units=512, activation="relu", input_shape=(784,)))
model.add(Dense(units=512, activation="relu"))
model.add(Dense(units=10, activation="softmax"))
model.summary()
model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid))
