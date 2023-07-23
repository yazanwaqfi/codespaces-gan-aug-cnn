import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import glob
import cv2
import os


def get_images(images_path):
    addresses = glob.glob(images_path)
    images_list = list(map(cv2.imread, addresses))
    return images_list


def get_images_gryscaled(images_path):
    addresses = glob.glob(images_path)
    images_list = list(map(imread_grayscale, addresses))
    return images_list


def get_labels(labels_path):
    with open(labels_path) as f:
        labels = f.readlines()
    for i in range(len(labels)):
        labels[i] = int(labels[i].strip('\n'))
    return labels


def imread_grayscale(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def show_image(image):
    plt.imshow(image)
    plt.show()


def resize_images(images_list, size):
    new_images = cv2.resize(images_list, (size, size))
    return new_images


def resize_images_2(images_list, size):
    new_images_list = []
    for i in range(len(images_list)):
        new_images_list.append(cv2.resize(images_list[i], (size, size)))
    return new_images_list


#  "train images/*.png"  "train labels.txt"
def get_dataset_ready(images_path, labels_path):
    images = get_images(images_path)
    labels = get_labels(labels_path)
    #r_images = resize_images_2(images, 28)
    new_images = np.array(images)
    new_labels = np.array(labels)

    return new_images, new_labels, images


def get_apti_dataset():
    pass


# print(labels)
# show_image(images[5])

train_images, train_labels, original_images = get_dataset_ready("train images/*.png", "train labels.txt")
test_images, test_labels, t_o_i = get_dataset_ready("test images/*.png", "test labels.txt")

train_labels = train_labels - 1
test_labels = test_labels - 1


print(train_labels)
print(train_images[5].shape)

# print(len(testing_label))
# print(len(train_labels))

# to reshape the image and normalize the pixls values of images
train_images = train_images.reshape(-1, 32, 32, 3)  # train_images.shape[0]
train_images = train_images / 255.0

test_images = test_images.reshape(-1, 32, 32, 3)  # train_images.shape[0]
test_images = test_images / 255.0

# split dataset tp smaller sizes (patch size and shuffle the images)
BUFFER_SIZE = train_images.shape[0]
BATCH_SIZE = 40

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# Discriminator Model
def make_discriminator_model():
    model = tf.keras.Sequential()
    # 7 filters
    model.add(tf.keras.layers.Conv2D(7, (3, 3), padding="same", input_shape=(32, 32, 1)))
    # flatten (we need flatten to move from conv to dinse layers)
    model.add(tf.keras.layers.Flatten())
    # LeakyReLU for add some non-leanyarity
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(50, activation="relu"))
    model.add(tf.keras.layers.Dense(29))
    return model


model_discriminator = make_discriminator_model()
model_discriminator(np.random.rand(1, 32, 32, 1).astype("float32"))
discriminator_optimizer = tf.optimizers.Adam(1e-3)


def get_discriminator_loss(real_predictions, fake_predictions):
    real_predictions = tf.sigmoid(real_predictions)
    fake_predictions = tf.sigmoid(fake_predictions)
    real_loss = tf.losses.binary_crossentropy(tf.ones_like(real_predictions), real_predictions)
    fake_loss = tf.losses.binary_crossentropy(tf.ones_like(fake_predictions), fake_predictions)
    return fake_loss + real_loss


# Generator
def make_generator_model():
    model = tf.keras.Sequential()
    # 7 width 7 hight and 256 fillters and input_shape 100 digits long
    model.add(tf.keras.layers.Dense(7 * 7 * 256, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    # convlotion transpose = inverse conv
    model.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding="same"))
    return model


generator = make_generator_model()
generator_optimizer = tf.optimizers.Adam(1e-4)


def get_generator_loss(fake_predictions):
    fake_predictions = tf.sigmoid(fake_predictions)
    fake_loss = tf.losses.binary_crossentropy(tf.ones_like(fake_predictions), fake_predictions)
    return fake_loss


# Training
def train(dataset, epochs):
    for _ in range(epochs):
        for images in dataset:
            images = tf.cast(images, tf.dtypes.float32)
            train_step(images)


def train_step(images):
    fake_image_noise = np.random.randn(BATCH_SIZE, 100).astype("float32")
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(fake_image_noise)
        real_output = model_discriminator(images)
        fake_output = model_discriminator(generated_images)

        gen_loss = get_generator_loss(fake_output)
        disc_loss = get_discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, model_discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, model_discriminator.trainable_variables))

        print("******** generator loss : ", np.mean(gen_loss))
        print("******** discriminator loss : ", np.mean(disc_loss))



