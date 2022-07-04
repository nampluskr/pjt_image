import sys
sys.path.insert(0, '..') ## '../..' for parent-parent directory

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

import os
import argparse
import numpy as np
import random

from mnist import load_mnist, load_fashion_mnist
import tf_gan_trainer as my

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


@tf.function
def load_data(image, label):
    image = tf.expand_dims(image, axis=-1)
    image = tf.cast(image, dtype=tf.float32)/255.
    label = tf.cast(label, dtype=tf.int64)
    return image, label


def get_dataloader(dataset, batch_size, training=False):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataloader = tf.data.Dataset.from_tensor_slices(dataset)
    dataloader = dataloader.map(load_data, num_parallel_calls=AUTOTUNE)
    if training:
        dataloader = dataloader.shuffle(1000)
    dataloader = dataloader.batch(batch_size).prefetch(AUTOTUNE)
    return dataloader


def cgan_mlp_generator():
    noises = layers.Input(shape=(100,))
    labels = layers.Input(shape=(1,))
    labels_ = tf.reshape(tf.one_hot(tf.cast(labels, tf.int64), depth=10), shape=(-1, 10))
    inputs = layers.Concatenate()([noises, labels_])

    model = keras.models.Sequential([
        layers.Input(shape=(100 + 10)),
        layers.Dense(256),
        layers.LeakyReLU(0.2),
        layers.LayerNormalization(),
        layers.Dense(28*28),
        layers.Reshape((28, 28, 1)),
        layers.Activation(tf.sigmoid),])

    outputs = model(inputs)
    return keras.Model(inputs=[noises, labels], outputs=outputs)


def cgan_mlp_discriminator():
    images = keras.Input(shape=(28, 28, 1))
    images_ = layers.Flatten()(images)
    labels = layers.Input(shape=(1,))
    labels_ = tf.reshape(tf.one_hot(tf.cast(labels, tf.int64), depth=10), shape=(-1, 10))
    inputs = layers.Concatenate()([images_, labels_])

    model = keras.models.Sequential([
        layers.Input(shape=(28*28 + 10,)),
        layers.Dense(200),
        layers.LeakyReLU(0.2),
        layers.LayerNormalization(),
        layers.Dense(1),])

    outputs = model(inputs)
    return keras.Model(inputs=[images, labels], outputs=outputs)


def cgan_cnn_generator(embedding_dim=100):
    noises_reshape = keras.models.Sequential([
        layers.Input(shape=(100,)),
        layers.Dense(7*7*256, use_bias=False),
        layers.ReLU(),
        layers.Reshape((7, 7, 256)),])

    labels_reshape = keras.models.Sequential([
        layers.Input(shape=(1,)),
        layers.Embedding(10, embedding_dim),
        layers.Dense(7*7*1),
        layers.Reshape((7, 7, 1)),])

    model = keras.models.Sequential([
        layers.Input(shape=(7, 7, 256 + 1)),
        layers.Conv2DTranspose(128, (5, 5), strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(momentum=0.9),
        layers.ReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(momentum=0.9),
        layers.ReLU(),
        layers.Conv2DTranspose(1, (5, 5), strides=2, padding='same', use_bias=False),
        layers.Activation(tf.sigmoid),])

    noises = keras.Input(shape=(100,))
    labels = keras.Input(shape=(1,))

    noises_labels = layers.Concatenate()([noises_reshape(noises), labels_reshape(labels)])
    outputs = model(noises_labels)
    return keras.Model(inputs=[noises, labels], outputs=outputs)


def cgan_cnn_discriminator(embedding_dim=100):
    labels_reshape = keras.models.Sequential([
        layers.Input(shape=(1,)),
        layers.Embedding(10, embedding_dim),
        layers.Dense(28*28*1),
        layers.Reshape((28, 28, 1)),])

    model = keras.models.Sequential([
        layers.Input(shape=(28, 28, 1 + 1)),
        layers.Conv2D(64, (5, 5), strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, (5, 5), strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Conv2D(256, (5, 5), strides=1, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1),])

    images = layers.Input(shape=(28, 28, 1))
    labels = layers.Input(shape=(1,))

    images_labels = layers.Concatenate()([images, labels_reshape(labels)])
    outputs = model(images_labels)
    return keras.Model(inputs=[images, labels], outputs=outputs)


def make_noises_labels(noise_size, noise_dim):
    noises = tf.random.normal((noise_size, noise_dim))
    labels = tf.transpose(tf.reshape(tf.repeat(tf.range(10), 5), (-1, 5)))
    labels = tf.cast(tf.reshape(labels, -1), tf.int64)
    return noises, labels


if __name__ == "__main__":

    ## Parameters:
    p = argparse.ArgumentParser()
    p.add_argument("--image_shape", type=tuple, default=(28, 28, 1))
    p.add_argument("--fashion", action='store_const', const=True, default=False)
    p.add_argument("--mlp", action='store_const', const=True, default=False)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--noise_dim", type=int, default=100)
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--log_interval", type=int, default=2)
    p.add_argument("--log_dir", type=str, default="log_tf_mnist_cgan")
    args = p.parse_args()

    manual_seed = 42
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    tf.random.set_seed(manual_seed)

    ## Dataset and Data Loaders:
    if args.fashion:
        data_dir = '../../datasets/fashion_mnist'
        args.log_dir += "_fashion"
        train_data, valid_data, class_names = load_fashion_mnist(data_dir, download=True)
    else:
        data_dir = '../../datasets/mnist'
        train_data, valid_data, class_names = load_mnist(data_dir, download=True)

    train_loader = get_dataloader(train_data, args.batch_size, training=True)
    valid_loader = get_dataloader(valid_data, args.batch_size, training=False)

    ## Modeling and Training:
    G = cgan_mlp_generator() if args.mlp else cgan_cnn_generator()
    D = cgan_mlp_discriminator() if args.mlp else cgan_cnn_discriminator()
    args.log_dir += "_mlp" if args.mlp else "_cnn"

    cgan = my.CondGAN(G, D, noise_dim=args.noise_dim)
    cgan.compile(g_optim=keras.optimizers.Adam(learning_rate=1e-4),
                 d_optim=keras.optimizers.Adam(learning_rate=1e-4),
                 loss_fn=keras.losses.MeanSquaredError())

    sample_noises = make_noises_labels(noise_size=50, noise_dim=args.noise_dim)
    hist = my.train(cgan, train_loader, args, sample_inputs=sample_noises)
    my.plot_progress(hist, args)

    ## Evaluation:
    gen_weights = os.path.join(args.log_dir, args.log_dir + "_gen_weights.h5")
    dis_weights = os.path.join(args.log_dir, args.log_dir + "_dis_weights.h5")

    trained_G = cgan_mlp_generator() if args.mlp else cgan_cnn_generator()
    trained_D = cgan_mlp_discriminator() if args.mlp else cgan_cnn_discriminator()

    trained_G.load_weights(gen_weights)
    trained_D.load_weights(dis_weights)