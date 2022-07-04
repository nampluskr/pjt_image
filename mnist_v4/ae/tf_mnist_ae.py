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
import tf_ae_trainer as my

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
    outputs = layers.Dense(latent_dim)(x)
    return keras.Model(inputs, outputs, name="encoder")


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
    x = layers.Conv2D(32, (3, 3), strides=2, padding="same")(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (3, 3), strides=2, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.ReLU()(x)

    outputs = layers.Dense(latent_dim)(x)
    return keras.Model(inputs, outputs, name="encoder")


def get_cnn_decoder(latent_dim=2):
    inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(256)(inputs)
    x = layers.ReLU()(x)
    x = layers.Dense(7*7*64)(x)
    x = layers.ReLU()(x)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(1, (3, 3), strides=1, padding="same")(x)
    outputs = layers.Activation(tf.sigmoid)(x)
    return keras.Model(inputs, outputs, name="decoder")


if __name__ == "__main__":

    ## Parameters:
    p = argparse.ArgumentParser()
    p.add_argument("--image_shape", type=tuple, default=(28, 28, 1))
    p.add_argument("--fashion", action='store_const', const=True, default=False)
    p.add_argument("--mlp", action='store_const', const=True, default=False)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--early_stop", action='store_const', const=True, default=False)
    p.add_argument("--min_loss", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--log_dir", type=str, default="log_tf_mnist_ae")
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
    def binary_accuracy(y_true, y_pred):
        y_true, y_pred = tf.round(y_true), tf.round(y_pred)
        return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

    encoder = get_mlp_encoder() if args.mlp else get_cnn_encoder()
    decoder = get_mlp_decoder() if args.mlp else get_cnn_decoder()
    args.log_dir += "_mlp" if args.mlp else "_cnn"

    ae = my.AutoEncoder(encoder, decoder)
    ae.compile(optim=keras.optimizers.Adam(),
               loss_fn=keras.losses.MeanSquaredError(),
               metric_fn=binary_accuracy, metric_name="acc")

    hist = my.train_with_metric(ae, train_loader, valid_loader, args)
    my.plot_progress(hist, args)

    ## Evaluation:
    encoder_weights = os.path.join(args.log_dir, args.log_dir + "_encoder_weights.h5")
    decoder_weights = os.path.join(args.log_dir, args.log_dir + "_decoder_weights.h5")
    trained_encoder = get_mlp_encoder() if args.mlp else get_cnn_encoder()
    trained_decoder = get_mlp_decoder() if args.mlp else get_cnn_decoder()
    trained_encoder.load_weights(encoder_weights)
    trained_decoder.load_weights(decoder_weights)

    trained_ae = my.AutoEncoder(trained_encoder, trained_decoder)
    trained_ae.compile(optim=keras.optimizers.Adam(),
               loss_fn=keras.losses.MeanSquaredError(),
               metric_fn=binary_accuracy, metric_name="acc")

    loss, acc = my.evaluate(trained_ae, valid_loader)
    print("\nTest: loss=%.4f, acc=%.4f" % (loss, acc))