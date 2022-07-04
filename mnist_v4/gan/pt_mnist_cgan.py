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


class CGanMlpGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, 28*28),
            nn.Unflatten(dim=1, unflattened_size=(1, 28, 28)),
            nn.Sigmoid(),)

    def forward(self, noises_labels):
        noises, labels = noises_labels
        labels_ = nn.functional.one_hot(labels, num_classes=10)
        inputs = torch.cat((noises, labels_), dim=-1)
        return self.model(inputs)


class CGanMlpDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28 + 10, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, 1),)

    def forward(self, images_labels):
        images, labels = images_labels
        images_ = nn.Flatten()(images)
        labels_ = nn.functional.one_hot(labels, num_classes=10)
        inputs = torch.cat((images_, labels_), dim=-1)
        return self.model(inputs)


class CGanCnnGenerator(nn.Module):
    def __init__(self, embedding_dim=100):
        super().__init__()
        self.noises_reshape = nn.Sequential(
            nn.Linear(100, 256*7*7, bias=False),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(256, 7, 7)),)

        self.labels_reshape = nn.Sequential(
            nn.Embedding(10, embedding_dim),
            nn.Linear(embedding_dim, 1*7*7),
            nn.Unflatten(dim=1, unflattened_size=(1, 7, 7)),)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(256 + 1, 128, (3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, (4, 4), stride=2, padding=1, bias=False),
            nn.Sigmoid(),)

    def forward(self, noises_labels):
        noises, labels = noises_labels
        noises = self.noises_reshape(noises)
        labels = self.labels_reshape(labels)
        inputs = torch.cat((noises, labels), dim=1)
        return self.model(inputs)


class CGanCnnDiscriminator(nn.Module):
    def __init__(self, embedding_dim=100):
        super().__init__()
        self.labels_reshape = nn.Sequential(
            nn.Embedding(10, embedding_dim),
            nn.Linear(embedding_dim, 1*28*28),
            nn.Unflatten(dim=1, unflattened_size=(1, 28, 28)),)

        self.model = nn.Sequential(
            nn.Conv2d(1 + 1, 64, (4, 4), stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, (4, 4), stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, (3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(256*7*7, 1),)

    def forward(self, images_labels):
        images, labels = images_labels
        labels = self.labels_reshape(labels)
        inputs = torch.cat((images, labels), dim=1)
        return self.model(inputs)


def make_noises_labels(noise_size, noise_dim):
    noises = torch.randn(noise_size, noise_dim)
    labels = torch.arange(10).repeat(5, 1).flatten().long()
    return noises.to(device), labels.to(device)


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
    p.add_argument("--log_dir", type=str, default="log_pt_mnist_cgan")
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
        G = CGanMlpGenerator().to(device)
        D = CGanMlpDiscriminator().to(device)
        args.log_dir += "_mlp"
    else:
        G = CGanCnnGenerator().to(device)
        D = CGanCnnDiscriminator().to(device)
        args.log_dir += "_cnn"

    cgan = my.CondGAN(G, D, noise_dim=args.noise_dim)
    cgan.compile(g_optim=torch.optim.Adam(G.parameters(), lr=1e-4),
                 d_optim=torch.optim.Adam(D.parameters(), lr=1e-4),
                 loss_fn=nn.BCEWithLogitsLoss())

    sample_noises = make_noises_labels(noise_size=50, noise_dim=args.noise_dim)
    hist = my.train(cgan, train_loader, args, sample_noises)
    my.plot_progress(hist, args)

    ## Evaluation:
    gen_weights = os.path.join(args.log_dir, args.log_dir + "_gen_weights.pth")
    dis_weights = os.path.join(args.log_dir, args.log_dir + "_dis_weights.pth")

    if args.mlp:
        trained_G = CGanMlpGenerator().to(device)
        trained_D = CGanMlpDiscriminator().to(device)
    else:
        trained_G = CGanCnnGenerator().to(device)
        trained_D = CGanCnnDiscriminator().to(device)

    trained_G.load_state_dict(torch.load(gen_weights))
    trained_D.load_state_dict(torch.load(dis_weights))