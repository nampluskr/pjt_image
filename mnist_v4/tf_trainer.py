import os
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from copy import deepcopy
import pathlib


class TrainerWithMetric(keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def compile(self, optim, loss_fn, metric_fn, metric_name):
        self.optim = optim
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.metric_name = metric_name

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss_fn(y, y_pred)
            metric = self.metric_fn(y, y_pred)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optim.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss, self.metric_name: metric}

    @tf.function
    def test_step(self, data):
        x, y = data
        y_pred = self.model(x, training=False)
        loss = self.loss_fn(y, y_pred)
        metric = self.metric_fn(y, y_pred)
        return {"loss": loss, self.metric_name: metric}


class TrainerWithMetric(keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def compile(self, optim, loss_fn, metric_fn, metric_name):
        self.optim = optim
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.metric_name = metric_name

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss_fn(y, y_pred)
            metric = self.metric_fn(y, y_pred)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optim.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss, self.metric_name: metric}

    @tf.function
    def test_step(self, data):
        x, y = data
        y_pred = self.model(x, training=False)
        loss = self.loss_fn(y, y_pred)
        metric = self.metric_fn(y, y_pred)
        return {"loss": loss, self.metric_name: metric}


class Trainer(keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def compile(self, optim, loss_fn):
        self.optim = optim
        self.loss_fn = loss_fn

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss_fn(y, y_pred)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optim.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}


def train_with_metric(trainer, train_loader, valid_loader, args):
    log_path = os.path.join(os.getcwd(), args.log_dir)
    if not pathlib.Path(log_path).exists():
        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(log_path, args.log_dir + ".txt")
    print_parameters(args, log_file)

    metric_name = trainer.metric_name
    hist = {'loss':[], metric_name:[], 'val_loss':[], 'val_' + metric_name:[]}
    best_loss, counter = 1e12, 1
    for epoch in range(args.n_epochs):

        ## Training
        desc = "Epoch[%3d/%3d]" % (epoch+1, args.n_epochs)
        with tqdm(train_loader, total=len(train_loader), ncols=100,
                  file=sys.stdout, ascii=True, leave=False) as pbar:
            pbar.set_description(desc)

            loss, metric = np.asfarray([]), np.asfarray([])
            for data in pbar:
                results = trainer.train_step(data)
                loss = np.append(loss, results['loss'].numpy())
                metric = np.append(metric, results[metric_name].numpy())
                pbar.set_postfix({'loss': "%.4f" % loss.mean(),
                                  metric_name: "%.4f" % metric.mean()})

            hist['loss'].append(loss.mean())
            hist[metric_name].append(metric.mean())

        ## Validation
        with tqdm(valid_loader, total=len(valid_loader), ncols=100,
                  file=sys.stdout, ascii=True, leave=False) as pbar:
            pbar.set_description(desc)
            
            val_loss, val_metric = np.asfarray([]), np.asfarray([])
            for data in pbar:
                results = trainer.test_step(data)
                val_loss = np.append(val_loss, results['loss'].numpy())
                val_metric = np.append(val_metric, results[metric_name].numpy())
                pbar.set_postfix({'val_loss': "%.4f" % val_loss.mean(),
                                  'val_' + metric_name: "%.4f" % val_metric.mean()})

            hist['val_loss'].append(val_loss.mean())
            hist['val_' + metric_name].append(val_metric.mean())

        if val_loss.mean() < best_loss and (best_loss - val_loss.mean()) > args.min_loss:
            best_loss = val_loss.mean()
            best_epoch = epoch + 1
            best_model = deepcopy(trainer.model.get_weights())
            counter = 1
        else:
            counter += 1

        ## Print log
        if (epoch + 1) % args.log_interval == 0 or args.early_stop:
            desc += ": loss=%.4f, %s=%.4f" % (loss.mean(), metric_name, metric.mean())
            desc += " - val_loss=%.4f, val_%s=%.4f (%d)" % (
                    val_loss.mean(), metric_name, val_metric.mean(), counter)
            print_log(desc, log_file)

        ## Early stopping
        if args.early_stop and counter == args.patience:
            print_log("Early stopped! (Best epoch=%d)" % best_epoch, log_file)
            break

    trainer.model.set_weights(best_model)
    trainer.model.save_weights(os.path.join(log_path, args.log_dir + "_weights.h5"))
    return hist


def train(trainer, train_loader, args):
    log_path = os.path.join(os.getcwd(), args.log_dir)
    if not pathlib.Path(log_path).exists():
        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(log_path, args.log_dir + ".txt")
    print_parameters(args, log_file)

    metric_name = trainer.metric_name
    hist = {'loss':[], metric_name:[]}
    for epoch in range(args.n_epochs):

        ## Training
        desc = "Epoch[%3d/%3d]" % (epoch+1, args.n_epochs)
        with tqdm(train_loader, total=len(train_loader), ncols=100,
                  file=sys.stdout, ascii=True, leave=False) as pbar:
            pbar.set_description(desc)

            loss = np.asfarray([])
            for data in pbar:
                results = trainer.train_step(data)
                loss = np.append(loss, results['loss'].numpy())
                pbar.set_postfix({'loss': "%.4f" % loss.mean(),})

            hist['loss'].append(loss.mean())

        ## Print log
        if (epoch + 1) % args.log_interval == 0:
            desc += ": loss=%.4f" % loss.mean()
            print_log(desc, log_file)

    trainer.model.save_weights(os.path.join(log_path, args.log_dir + "_weights.h5"))
    return hist


def evaluate(trainer, test_loader):
    with tqdm(test_loader, total=len(test_loader), ncols=100,
                  file=sys.stdout, ascii=True, leave=False) as pbar:

        loss, metric = np.asfarray([]), np.asfarray([])
        for data in pbar:
            results = trainer.test_step(data)

            loss = np.append(loss, results['loss'].numpy())
            metric = np.append(metric, results[trainer.metric_name].numpy())
            pbar.set_postfix({'loss': "%.4f" % loss.mean(),
                              trainer.metric_name: "%.4f" % metric.mean()})
    return loss.mean(), metric.mean()


def print_parameters(args, log_file):
    parameters = ""
    for key, value in vars(args).items():
        parameters += "%s=%s, " % (key, str(value))
    print(parameters[:-2] + '\n')

    with open(log_file, 'w') as f:
        f.write(parameters[:-2] + '\n\n')


def print_log(desc, log_file):
    print(desc)
    with open(log_file, 'a') as f:
        f.write(desc + '\n')


def plot_progress(hist, args, skip=1):
    fig, ax = plt.subplots(figsize=(8,4))
    for name, loss in hist.items():
        iter = range(1, len(loss) + 1)
        ax.plot(iter[::skip], loss[::skip], 'o-', label=name)
    ax.set_title(args.log_dir, fontsize=15)
    ax.set_xlabel("Epochs", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(color='k', ls=':', lw=1)
    fig.tight_layout()

    img_name = os.path.join(os.getcwd(), args.log_dir, args.log_dir + "_hist.png")
    plt.savefig(img_name, pad_inches=0)
    plt.close()


if __name__ == "__main__":

    pass