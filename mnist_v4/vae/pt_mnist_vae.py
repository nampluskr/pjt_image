import sys
sys.path.insert(0, '..') ## '../..' for parent-parent directory

import torch
import torch.nn as nn

import os
import argparse
import numpy as np
import random

from mnist import load_mnist, load_fashion_mnist
import pt_vae_trainer as my


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.images, self.labels = data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.tensor(image).unsqueeze(dim=0).float()/255.
        label = self.labels[idx]
        label = torch.tensor(label).long()
        return image, label


def get_dataloader(data, batch_size, training=True, use_cuda=False):
    kwargs = {'num_workers': 12, 'pin_memory': True} if use_cuda else {}
    dataloader = torch.utils.data.DataLoader(dataset=Dataset(data),
                                             batch_size=batch_size,
                                             shuffle=training, **kwargs)
    return dataloader


class MlpEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),)
        self.mean = nn.Linear(256, latent_dim)
        self.log_var = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.model(x)
        return self.mean(h), self.log_var(h)


class MlpDecoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid(),
            nn.Unflatten(dim=1, unflattened_size=(1, 28, 28)),)

    def forward(self, z):
        return self.model(z)


class CnnEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 256),
            nn.ReLU(),)
        self.mean = nn.Linear(256, latent_dim)
        self.log_var = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.model(x)
        return self.mean(h), self.log_var(h)


class CnnDecoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64*7*7),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(64, 7, 7)),
            nn.ConvTranspose2d(64, 64, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, (3, 3), stride=1, padding=1),
            nn.Sigmoid(),)

    def forward(self, z):
        return self.model(z)


if __name__ == "__main__":

    ## Parameters:
    p = argparse.ArgumentParser()
    p.add_argument("--image_shape", type=tuple, default=(1, 28, 28))
    p.add_argument("--fashion", action='store_const', const=True, default=False)
    p.add_argument("--mlp", action='store_const', const=True, default=False)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--cpu", action='store_const', const=True, default=False)
    p.add_argument("--log_dir", type=str, default="log_pt_mnist_vae")
    args = p.parse_args()

    manual_seed = 42
    random.seed(manual_seed)
    np.random.seed(manual_seed)

    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        torch.cuda.manual_seed(manual_seed)
    else:
        torch.manual_seed(manual_seed)

    ## Dataset and Data Loaders:
    if args.fashion:
        data_path = '../../datasets/fashion_mnist'
        args.log_dir += "_fashion"
        train_data, valid_data, class_names = load_fashion_mnist(data_path, download=True)
    else:
        data_path = '../../datasets/mnist'
        train_data, valid_data, class_names = load_mnist(data_path, download=True)

    train_loader = get_dataloader(train_data, args.batch_size, training=True,
                                  use_cuda=use_cuda)
    valid_loader = get_dataloader(valid_data, args.batch_size, training=False,
                                 use_cuda=use_cuda)

    ## Modeling and Training:
    def loss_fn_bce(x_pred, x):
        bce = nn.functional.binary_cross_entropy(x_pred, x, reduction='sum')
        return bce

    def loss_fn_kld(z_mean, z_log_var):
        kld = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return kld

    def binary_accuracy(x_pred, x_true):
        return torch.eq(x_pred.round(), x_true.round()).float().mean()

    encoder = MlpEncoder().to(device) if args.mlp else CnnEncoder().to(device)
    decoder = MlpDecoder().to(device) if args.mlp else CnnDecoder().to(device)
    args.log_dir += "_mlp" if args.mlp else "_cnn"

    vae = my.VAE(encoder, decoder)
    vae.compile(optim=torch.optim.Adam(vae.parameters()),
               loss_fn_bce=nn.BCELoss(reduction="sum"),
               loss_fn_kld=loss_fn_kld,
               metric_fn=binary_accuracy, metric_name="acc")

    hist = my.train(vae, train_loader, valid_loader, args)
    my.plot_progress(hist, args)

    ## Evaluation:
    encoder_weights = os.path.join(args.log_dir, args.log_dir + "_encoder_weights.pth")
    decoder_weights = os.path.join(args.log_dir, args.log_dir + "_decoder_weights.pth")
    trained_encoder = MlpEncoder().to(device) if args.mlp else CnnEncoder().to(device)
    trained_decoder = MlpDecoder().to(device) if args.mlp else CnnDecoder().to(device)
    trained_encoder.load_state_dict(torch.load(encoder_weights))
    trained_decoder.load_state_dict(torch.load(decoder_weights))

    trained_vae = my.VAE(trained_encoder, trained_decoder)
    trained_vae.compile(optim=torch.optim.Adam(vae.parameters()),
               loss_fn_bce=nn.BCELoss(reduction="sum"),
               loss_fn_kld=loss_fn_kld,
               metric_fn=binary_accuracy, metric_name="acc")

    bce, kld, acc = my.evaluate(trained_vae, valid_loader)
    print("\nTest: bce=%.f, kld=%.f, acc=%.4f" % (bce, kld, acc))