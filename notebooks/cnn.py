import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

train_images = 'get data of ......'
train_labels = 'get the data here....'


test_images = 'get data of .....'
test_labels = 'get the data here....'

modelf = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(28, activation="softmax")
])

conv_model = tf.keras.Sequential()
conv_model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding="same", input_shape=train_images.shape[1:]))
conv_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

conv_model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation=tf.nn.leaky_relu))
conv_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

conv_model.add(tf.keras.layers.Flatten())
conv_model.add(tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu))
conv_model.add(tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu))
# conv_model.add(tf.keras.layers.Dropout(0.2))
conv_model.add(tf.keras.layers.Dense(28, activation="softmax"))  # "sigmoid"

conv_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
conv_model.fit(train_images, train_labels, epochs=100)
tes_loss, test_acc = conv_model.evaluate(test_images, test_labels)

print("\n\n******** Tested Accuracy : ", test_acc)
