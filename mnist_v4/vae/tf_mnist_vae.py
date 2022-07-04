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
import tf_vae_trainer as my

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


def get_mlp_encoder(latent_dim=2):
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Flatten()(inputs)
    x = layers.Dense(256)(x)
    x = layers.ReLU()(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    return keras.Model(inputs, outputs=[z_mean, z_log_var], name="encoder")


def get_mlp_decoder(latent_dim=2):
    inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(256)(inputs)
    x = layers.ReLU()(x)
    x = layers.Dense(28*28)(x)
    x = layers.Activation(tf.sigmoid)(x)
    outputs = layers.Reshape((28, 28, 1))(x)
    return keras.Model(inputs, outputs, name="decoder")


def get_cnn_encoder(latent_dim=2):
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, (3, 3), strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    return keras.Model(inputs, [z_mean, z_log_var], name="encoder")


def get_cnn_decoder(latent_dim=2):
    inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dense(7*7*64, activation="relu")(x)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2DTranspose(1, (3, 3), strides=1, padding="same", activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="decoder")


if __name__ == "__main__":

    ## Parameters:
    p = argparse.ArgumentParser()
    p.add_argument("--image_shape", type=tuple, default=(28, 28, 1))
    p.add_argument("--fashion", action='store_const', const=True, default=False)
    p.add_argument("--mlp", action='store_const', const=True, default=False)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--early_stop", action='store_const', const=True, default=False)
    p.add_argument("--min_loss", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--log_dir", type=str, default="log_tf_mnist_vae")
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
    @tf.function
    def loss_fn_bce(x, x_pred):
        bce = keras.losses.binary_crossentropy(x, x_pred)
        return tf.reduce_sum(bce)

    @tf.function
    def loss_fn_kld(z_mean, z_log_var):
        kld = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        return kld

    @tf.function
    def binary_accuracy(y_true, y_pred):
        y_true, y_pred = tf.round(y_true), tf.round(y_pred)
        return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

    encoder = get_mlp_encoder() if args.mlp else get_cnn_encoder()
    decoder = get_mlp_decoder() if args.mlp else get_cnn_decoder()
    args.log_dir += "_mlp" if args.mlp else "_cnn"

    vae = my.VAE(encoder, decoder)
    vae.compile(optim=keras.optimizers.Adam(),
               loss_fn_bce=loss_fn_bce,
               loss_fn_kld=loss_fn_kld,
               metric_fn=binary_accuracy, metric_name="acc")

    hist = my.train(vae, train_loader, valid_loader, args)
    my.plot_progress(hist, args)

    ## Evaluation:
    encoder_weights = os.path.join(args.log_dir, args.log_dir + "_encoder_weights.h5")
    decoder_weights = os.path.join(args.log_dir, args.log_dir + "_decoder_weights.h5")
    trained_encoder = get_mlp_encoder() if args.mlp else get_cnn_encoder()
    trained_decoder = get_mlp_decoder() if args.mlp else get_cnn_decoder()
    trained_encoder.load_weights(encoder_weights)
    trained_decoder.load_weights(decoder_weights)

    trained_vae = my.VAE(trained_encoder, trained_decoder)
    trained_vae.compile(optim=keras.optimizers.Adam(),
               loss_fn_bce=loss_fn_bce,
               loss_fn_kld=loss_fn_kld,
               metric_fn=binary_accuracy, metric_name="acc")

    bce, kld, acc = my.evaluate(trained_vae, valid_loader)
    print("\nTest: bce=%.f, kld=%.f, acc=%.4f" % (bce, kld, acc))