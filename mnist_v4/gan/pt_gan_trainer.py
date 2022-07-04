import os
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

import sys
from tqdm import tqdm
from copy import deepcopy
import pathlib


class GAN(nn.Module):
    def __init__(self, generator, discriminator, noise_dim):
        super().__init__()
        self.G = generator
        self.D = discriminator
        self.noise_dim = noise_dim
        self.device = next(self.G.parameters()).device

    def compile(self, g_optim, d_optim, loss_fn):
        self.g_optim = g_optim
        self.d_optim = d_optim
        self.loss_fn = loss_fn

    def train_step(self, inputs):
        r_images, _ = inputs

        ## Train the discriminator:
        noises = torch.randn((r_images.shape[0], self.noise_dim)).to(self.device)
        f_images = self.G(noises).detach()

        r_logits = self.D(r_images)
        f_logits = self.D(f_images)

        r_loss = self.loss_fn(r_logits, torch.ones_like(r_logits))
        f_loss = self.loss_fn(f_logits, torch.zeros_like(f_logits))
        d_loss = r_loss + f_loss

        self.d_optim.zero_grad()
        d_loss.backward()
        self.d_optim.step()

        ## Train the generator:
        noises = torch.randn((r_images.shape[0], self.noise_dim)).to(self.device)
        g_images = self.G(noises)
        g_logits = self.D(g_images)
        g_loss = self.loss_fn(g_logits, torch.ones_like(g_logits))

        self.g_optim.zero_grad()
        g_loss.backward()
        self.g_optim.step()
        return {'d_loss':d_loss, 'g_loss':g_loss}


class CondGAN(GAN):
        
    def train_step(self, inputs):
        r_images, labels = inputs

        ## Train the discriminator:
        noises = torch.randn((r_images.shape[0], self.noise_dim)).to(self.device)
        f_images = self.G([noises, labels]).detach()

        r_logits = self.D([r_images, labels])
        f_logits = self.D([f_images, labels])

        r_loss = self.loss_fn(r_logits, torch.ones_like(r_logits))
        f_loss = self.loss_fn(f_logits, torch.zeros_like(f_logits))
        d_loss = r_loss + f_loss

        self.d_optim.zero_grad()
        d_loss.backward()
        self.d_optim.step()

        ## Train the generator:
        noises = torch.randn((r_images.shape[0], self.noise_dim)).to(self.device)
        g_images = self.G([noises, labels])
        g_logits = self.D([g_images, labels])
        g_loss = self.loss_fn(g_logits, torch.ones_like(g_logits))

        self.g_optim.zero_grad()
        g_loss.backward()
        self.g_optim.step()
        return {'d_loss':d_loss, 'g_loss':g_loss}


class AuxCondGAN(GAN):
    def compile(self, g_optim, d_optim, loss_fn, loss_fn_aux):
        super().compile(g_optim, d_optim, loss_fn)
        self.loss_fn_aux = loss_fn_aux

    def train_step(self, inputs):
        r_images, labels = inputs

        ## Train the discriminator:
        noises = torch.randn((r_images.shape[0], self.noise_dim)).to(self.device)
        f_images = self.G([noises, labels]).detach()

        r_logits, r_preds = self.D([r_images, labels])
        f_logits, f_preds = self.D([f_images, labels])

        r_loss = self.loss_fn(r_logits, torch.ones_like(r_logits))
        f_loss = self.loss_fn(f_logits, torch.zeros_like(f_logits))

        d_loss = r_loss + f_loss
        d_loss += self.loss_fn_aux(r_preds, labels)
        d_loss += self.loss_fn_aux(f_preds, labels)

        self.d_optim.zero_grad()
        d_loss.backward()
        self.d_optim.step()

        ## Train the generator:
        noises = torch.randn((r_images.shape[0], self.noise_dim)).to(self.device)
        g_images = self.G([noises, labels])
        g_logits, g_preds = self.D([g_images, labels])
        g_loss = self.loss_fn(g_logits, torch.ones_like(g_logits))
        g_loss += self.loss_fn_aux(g_preds, labels)

        self.g_optim.zero_grad()
        g_loss.backward()
        self.g_optim.step()
        return {'d_loss':d_loss, 'g_loss':g_loss}


def train(gan, train_loader, args, sample_inputs):
    log_path = os.path.join(os.getcwd(), args.log_dir)
    if not pathlib.Path(log_path).exists():
        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(log_path, args.log_dir + ".txt")
    print_parameters(args, log_file)

    hist = {'d_loss':[], 'g_loss':[]}
    for epoch in range(args.n_epochs):
        desc = "Epoch[%3d/%3d]" % (epoch+1, args.n_epochs)
        with tqdm(train_loader, total=len(train_loader), ncols=100,
                file=sys.stdout, ascii=True, leave=False) as pbar:
            pbar.set_description(desc)

            d_loss, g_loss = np.asfarray([]), np.asfarray([])
            for x, y in pbar:
                inputs = x.to(gan.device), y.to(gan.device)
                results = gan.train_step(inputs)
                
                d_loss = np.append(d_loss, results['d_loss'].item())
                g_loss = np.append(g_loss, results['g_loss'].item())
                pbar.set_postfix(d_loss="%.3f" % d_loss.mean(),
                                 g_loss="%.3f" % g_loss.mean())

            hist['d_loss'].append(d_loss.mean())
            hist['g_loss'].append(g_loss.mean())

        print_log(desc + ": d_loss=%.3f, g_loss=%.3f" % (
            d_loss.mean(), g_loss.mean()), log_file)

        if (epoch + 1) % args.log_interval == 0:
            sample_images = gan.G(sample_inputs).cpu().detach().numpy()
            sample_images = sample_images.reshape(-1, 28, 28)*255
            img_name = os.path.join(log_path, args.log_dir + "-%03d.png" % (epoch+1))
            save_images(sample_images, img_name=img_name)

    torch.save(gan.G.state_dict(), os.path.join(log_path, args.log_dir + "_gen_weights.pth"))
    torch.save(gan.D.state_dict(), os.path.join(log_path, args.log_dir + "_dis_weights.pth"))
    return hist


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


def save_images(images, labels=None, n_cols=10, width=8, img_name=""):
    n_rows = images.shape[0] // n_cols + (1 if images.shape[0] % n_cols else 0)
    height = width*n_rows/n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height))
    for i, ax in enumerate(axes.flat):
        if i < images.shape[0]:
            ax.imshow(images[i].astype('uint8'), interpolation='none', cmap='gray_r')
            if labels is not None:
                ax.set_title(labels[i])
        ax.set_axis_off()
    fig.tight_layout()

    if labels is None:
        plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(img_name, pad_inches=0)
    plt.close()


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