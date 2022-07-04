import os
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

import sys
from tqdm import tqdm
from copy import deepcopy
import pathlib


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = next(encoder.parameters()).device

    def compile(self, optim, loss_fn_bce, loss_fn_kld, metric_fn, metric_name):
        self.optim = optim
        self.loss_fn_bce = loss_fn_bce
        self.loss_fn_kld = loss_fn_kld
        self.metric_fn = metric_fn
        self.metric_name = metric_name

    def sampling(self, mean, log_var):
        epsilon = torch.randn_like(mean)
        return mean + torch.exp(0.5 * log_var) * epsilon

    def train_step(self, inputs):
        x, _ = inputs
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling(z_mean, z_log_var)
        x_pred = self.decoder(z)
        bce_loss = self.loss_fn_bce(x_pred, x)
        kld_loss = self.loss_fn_kld(z_mean, z_log_var)
        loss = bce_loss + kld_loss
        metric = self.metric_fn(x_pred, x)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {'bce': bce_loss, 'kld': kld_loss, self.metric_name:metric}

    @torch.no_grad()
    def test_step(self, inputs):
        x, _ = inputs
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling(z_mean, z_log_var)
        x_pred = self.decoder(z)
        bce_loss = self.loss_fn_bce(x_pred, x)
        kld_loss = self.loss_fn_kld(z_mean, z_log_var)
        metric = self.metric_fn(x_pred, x)
        return {'bce': bce_loss, 'kld': kld_loss, self.metric_name:metric}


class CondVAE(VAE):
    def train_step(self, inputs):
        x, y = inputs
        z_mean, z_log_var = self.encoder([x, y])
        z = self.sampling(z_mean, z_log_var)
        x_pred = self.decoder([z, y])
        bce_loss = self.loss_fn_bce(x_pred, x)
        kld_loss = self.loss_fn_kld(z_mean, z_log_var)
        loss = bce_loss + kld_loss
        metric = self.metric_fn(x_pred, x)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {'bce': bce_loss, 'kld': kld_loss, self.metric_name:metric}

    @torch.no_grad()
    def test_step(self, inputs):
        x, y = inputs
        z_mean, z_log_var = self.encoder([x, y])
        z = self.sampling(z_mean, z_log_var)
        x_pred = self.decoder([z, y])
        bce_loss = self.loss_fn_bce(x_pred, x)
        kld_loss = self.loss_fn_kld(z_mean, z_log_var)
        metric = self.metric_fn(x_pred, x)
        return {'bce': bce_loss, 'kld': kld_loss, self.metric_name:metric}


def train(vae, train_loader, valid_loader, args):
    log_path = os.path.join(os.getcwd(), args.log_dir)
    if not pathlib.Path(log_path).exists():
        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(log_path, args.log_dir + ".txt")
    print_parameters(args, log_file)

    metric_name = vae.metric_name
    hist = {'bce':[], 'kld':[], metric_name:[], 
            'val_bce':[], 'val_kld':[], 'val_' + metric_name:[]}
    for epoch in range(args.n_epochs):

        ## Training
        vae.train()
        desc = "Epoch[%3d/%3d]" % (epoch+1, args.n_epochs)
        with tqdm(train_loader, total=len(train_loader), ncols=100,
                file=sys.stdout, ascii=True, leave=False) as pbar:
            pbar.set_description(desc)

            bce, kld, metric = np.asfarray([]), np.asfarray([]), np.asfarray([])
            for x, y in pbar:
                inputs = x.to(vae.device), y.to(vae.device)
                results = vae.train_step(inputs)

                bce = np.append(bce, results['bce'].item())
                kld = np.append(kld, results['kld'].item())
                metric = np.append(metric, results[metric_name].item())
                pbar.set_postfix({'bce': "%.f" % bce.mean(),
                                  'kld': "%.f" % kld.mean(),
                                  metric_name: "%.4f" % metric.mean()})

            hist['bce'].append(bce.mean())
            hist['kld'].append(kld.mean())
            hist[metric_name].append(metric.mean())

        ## Validation
        vae.eval()
        with tqdm(valid_loader, total=len(valid_loader), ncols=100,
                file=sys.stdout, ascii=True, leave=False) as pbar:
            pbar.set_description(desc)

            val_bce, val_kld, val_metric = np.asfarray([]), np.asfarray([]), np.asfarray([])
            for x, y in pbar:
                inputs = x.to(vae.device), y.to(vae.device)
                results = vae.test_step(inputs)
                
                val_bce = np.append(val_bce, results['bce'].item())
                val_kld = np.append(val_kld, results['kld'].item())
                val_metric = np.append(val_metric, results[metric_name].item())
                pbar.set_postfix({'val_bce': "%.f" % val_bce.mean(),
                                  'val_kld': "%.f" % val_kld.mean(),
                                  'val_' + metric_name: "%.4f" % val_metric.mean()})

            hist['val_bce'].append(val_bce.mean())
            hist['val_kld'].append(val_kld.mean())
            hist['val_' + metric_name].append(val_metric.mean())

        ## Print log
        if (epoch + 1) % args.log_interval == 0 or args.early_stop:
            desc += ": bce=%.f, kld=%.f, %s=%.4f" % (bce.mean(), kld.mean(), metric_name, metric.mean())
            desc += " - val_bce=%.f, val_kld=%.f, val_%s=%.4f" % (
                    val_bce.mean(), val_kld.mean(), metric_name, val_metric.mean())
            print_log(desc, log_file)

    torch.save(vae.encoder.state_dict(), 
               os.path.join(log_path, args.log_dir + "_encoder_weights.pth"))
    torch.save(vae.decoder.state_dict(), 
               os.path.join(log_path, args.log_dir + "_decoder_weights.pth"))
    return hist


def evaluate(vae, test_loader):
    vae.eval()
    with tqdm(test_loader, total=len(test_loader), ncols=100,
        file=sys.stdout, ascii=True, leave=False) as pbar:

        bce, kld, metric = np.asfarray([]), np.asfarray([]), np.asfarray([])
        for x, y in pbar:
            inputs = x.to(vae.device), y.to(vae.device)
            results = vae.test_step(inputs)

            bce = np.append(bce, results['bce'].item())
            kld = np.append(kld, results['kld'].item())
            metric = np.append(metric, results[vae.metric_name].item())
            pbar.set_postfix({'bce': "%.4f" % bce.mean(),
                              'kld': "%.4f" % kld.mean(),
                              vae.metric_name: "%.4f" % metric.mean()})

    return bce.mean(), kld.mean(), metric.mean()


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