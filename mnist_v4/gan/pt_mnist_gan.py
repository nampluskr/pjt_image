import sys
sys.path.insert(0, '..') ## '../..' for parent-parent directory

import torch
import torch.nn as nn

import os
import argparse
import numpy as np
import random

from mnist import load_mnist, load_fashion_mnist
import pt_gan_trainer as my


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


def gan_mlp_generator():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(100, 256),
        nn.LeakyReLU(0.2),
        nn.LayerNorm(256),
        nn.Linear(256, 28*28),
        nn.Sigmoid(),
        nn.Unflatten(dim=1, unflattened_size=(1, 28, 28)),)
    return model


def gan_mlp_discriminator():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 256),
        nn.LeakyReLU(0.2),
        nn.LayerNorm(256),
        nn.Linear(256, 1),)
    return model


def gan_cnn_generator(n_channels=256):
    ch = n_channels
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(100, ch*7*7, bias=False),
        nn.ReLU(inplace=True),
        nn.Unflatten(dim=1, unflattened_size=(ch, 7, 7)),

        nn.ConvTranspose2d(ch, ch//2, (3, 3), stride=1, padding=1, bias=False),
        nn.BatchNorm2d(ch//2, momentum=0.9),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(ch//2, ch//4, (4, 4), stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64, momentum=0.9),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(ch//4, 1, (4, 4), stride=2, padding=1, bias=False),
        nn.Sigmoid(),)
    return model


def gan_cnn_discriminator(n_channels=256):
    ch = n_channels
    model = nn.Sequential(
        nn.Conv2d(1, ch//4, (4, 4), stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ch//4, ch//2, (4, 4), stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ch//2, ch, (3, 3), stride=1, padding=1),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Flatten(),
        nn.Linear(ch*7*7, 1),)
    return model


def make_noises(noise_size, noise_dim):
    noises = torch.randn(noise_size, noise_dim)
    return noises.to(device)


if __name__ == "__main__":

    ## Parameters:
    p = argparse.ArgumentParser()
    p.add_argument("--image_shape", type=tuple, default=(1, 28, 28))
    p.add_argument("--fashion", action='store_const', const=True, default=False)
    p.add_argument("--mlp", action='store_const', const=True, default=False)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--noise_dim", type=int, default=100)
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--log_interval", type=int, default=2)
    p.add_argument("--cpu", action='store_const', const=True, default=False)
    p.add_argument("--log_dir", type=str, default="log_pt_mnist_gan")
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
    if args.mlp:
        G = gan_mlp_generator().to(device)
        D = gan_mlp_discriminator().to(device)
        args.log_dir += "_mlp"
    else:
        G = gan_cnn_generator().to(device)
        D = gan_cnn_discriminator().to(device)
        args.log_dir += "_cnn"

    gan = my.GAN(G, D, noise_dim=args.noise_dim)
    gan.compile(g_optim=torch.optim.Adam(G.parameters(), lr=1e-4),
                d_optim=torch.optim.Adam(D.parameters(), lr=1e-4),
                loss_fn=nn.BCEWithLogitsLoss())

    sample_noises = make_noises(noise_size=50, noise_dim=args.noise_dim)
    hist = my.train(gan, train_loader, args, sample_noises)
    my.plot_progress(hist, args)

    ## Evaluation:
    gen_weights = os.path.join(args.log_dir, args.log_dir + "_gen_weights.pth")
    dis_weights = os.path.join(args.log_dir, args.log_dir + "_dis_weights.pth")

    if args.mlp:
        trained_G = gan_mlp_generator().to(device)
        trained_D = gan_mlp_discriminator().to(device)
    else:
        trained_G = gan_cnn_generator().to(device)
        trained_D = gan_cnn_discriminator().to(device)

    trained_G.load_state_dict(torch.load(gen_weights))
    trained_D.load_state_dict(torch.load(dis_weights))