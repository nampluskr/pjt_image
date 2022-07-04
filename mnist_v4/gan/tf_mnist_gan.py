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


def gan_mlp_generator():
    inputs = keras.Input(shape=(100,))
    x = layers.Dense(256)(inputs)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(28*28)(x)
    x = layers.Activation(tf.sigmoid)(x)
    outputs = layers.Reshape((28, 28, 1))(x)
    return keras.Model(inputs, outputs)


def gan_mlp_discriminator():
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Flatten()(inputs)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.LayerNormalization()(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)


def gan_cnn_generator(n_channels=256):
    ch = n_channels
    inputs = keras.Input(shape=(100,))
    model = keras.models.Sequential([
        layers.Dense(7*7*ch, use_bias=False),
        layers.ReLU(),
        layers.Reshape((7, 7, ch)),

        layers.Conv2DTranspose(ch//2, (5, 5), strides=1, padding='same', use_bias=False),
        layers.BatchNormalization(momentum=0.9),
        layers.ReLU(),

        layers.Conv2DTranspose(ch//4, (5, 5), strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(momentum=0.9),
        layers.ReLU(),

        layers.Conv2DTranspose(1, (5, 5), strides=2, padding='same', use_bias=False),
        layers.Activation(tf.sigmoid),])
    outputs = model(inputs)
    return keras.Model(inputs, outputs)


def gan_cnn_discriminator(n_channels=256):
    ch = n_channels
    inputs = keras.Input(shape=(28, 28, 1))
    model = keras.models.Sequential([
        layers.Conv2D(ch//4, (5, 5), strides=2, padding='same'),
        layers.LeakyReLU(0.2),

        layers.Conv2D(ch//2, (5, 5), strides=2, padding='same'),
        layers.LeakyReLU(0.2),

        layers.Conv2D(ch, (5, 5), strides=1, padding='same'),
        layers.LeakyReLU(0.2),

        layers.Flatten(),
        layers.Dense(1),])
    outputs = model(inputs)
    return keras.Model(inputs, outputs)


def make_noises(noise_size, noise_dim):
    noises = tf.random.normal((noise_size, noise_dim))
    return noises


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
    p.add_argument("--log_dir", type=str, default="log_tf_mnist_gan")
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
    G = gan_mlp_generator() if args.mlp else gan_cnn_generator()
    D = gan_mlp_discriminator() if args.mlp else gan_cnn_discriminator()
    args.log_dir += "_mlp" if args.mlp else "_cnn"

    gan = my.GAN(G, D, noise_dim=args.noise_dim)
    gan.compile(g_optim=keras.optimizers.Adam(learning_rate=1e-4),
                d_optim=keras.optimizers.Adam(learning_rate=1e-4),
                loss_fn=keras.losses.MeanSquaredError())

    sample_noises = make_noises(noise_size=50, noise_dim=args.noise_dim)
    hist = my.train(gan, train_loader, args, sample_inputs=sample_noises)
    my.plot_progress(hist, args)

    ## Evaluation:
    gen_weights = os.path.join(args.log_dir, args.log_dir + "_gen_weights.h5")
    dis_weights = os.path.join(args.log_dir, args.log_dir + "_dis_weights.h5")

    trained_G = gan_mlp_generator() if args.mlp else gan_cnn_generator()
    trained_D = gan_mlp_discriminator() if args.mlp else gan_cnn_discriminator()

    trained_G.load_weights(gen_weights)
    trained_D.load_weights(dis_weights)