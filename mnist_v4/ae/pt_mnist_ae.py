import sys
sys.path.insert(0, '..') ## '../..' for parent-parent directory

import torch
import torch.nn as nn

import os
import argparse
import numpy as np
import random

from mnist import load_mnist, load_fashion_mnist
import pt_ae_trainer as my


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


def get_mlp_encoder(latent_dim=2):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 256),
        nn.ReLU(),
        nn.Linear(256, latent_dim),)
    return model


def get_mlp_decoder(latent_dim=2):
    model = nn.Sequential(
        nn.Linear(latent_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 28*28),
        nn.Sigmoid(),
        nn.Unflatten(dim=1, unflattened_size=(1, 28, 28)),)
    return model


def get_cnn_encoder(latent_dim=2):
    model = nn.Sequential(
        nn.Conv2d(1, 32, (3, 3), stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, (3, 3), stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64*7*7, 256),
        nn.ReLU(),
        nn.Linear(256, latent_dim),)
    return model


def get_cnn_decoder(latent_dim=2):
    model = nn.Sequential(
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
    return model


if __name__ == "__main__":
    
    ## Parameters:
    p = argparse.ArgumentParser()
    p.add_argument("--image_shape", type=tuple, default=(1, 28, 28))
    p.add_argument("--fashion", action='store_const', const=True, default=False)
    p.add_argument("--mlp", action='store_const', const=True, default=False)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--cpu", action='store_const', const=True, default=False)
    p.add_argument("--early_stop", action='store_const', const=True, default=False)
    p.add_argument("--min_loss", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--log_dir", type=str, default="log_pt_mnist_ae")
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
    def binary_accuracy(x_pred, x_true):
        return torch.eq(x_pred.round(), x_true.round()).float().mean()
    
    encoder = get_mlp_encoder().to(device) if args.mlp else get_cnn_encoder().to(device)
    decoder = get_mlp_decoder().to(device) if args.mlp else get_cnn_decoder().to(device)
    args.log_dir += "_mlp" if args.mlp else "_cnn"

    ae = my.AutoEncoder(encoder, decoder)
    ae.compile(optim=torch.optim.Adam(ae.parameters()),
               loss_fn=nn.BCELoss(),
               metric_fn=binary_accuracy, metric_name="acc")

    hist = my.train_with_metric(ae, train_loader, valid_loader, args)
    my.plot_progress(hist, args)
    
    ## Evaluation:
    encoder_weights = os.path.join(args.log_dir, args.log_dir + "_encoder_weights.pth")
    decoder_weights = os.path.join(args.log_dir, args.log_dir + "_decoder_weights.pth")
    trained_encoder = get_mlp_encoder().to(device) if args.mlp else get_cnn_encoder().to(device)
    trained_decoder = get_mlp_decoder().to(device) if args.mlp else get_cnn_decoder().to(device)
    trained_encoder.load_state_dict(torch.load(encoder_weights))
    trained_decoder.load_state_dict(torch.load(decoder_weights))

    trained_ae = my.AutoEncoder(trained_encoder, trained_decoder)
    trained_ae.compile(optim=torch.optim.Adam(ae.parameters()),
               loss_fn=nn.MSELoss(),
            #    loss_fn=nn.BCELoss(),
               metric_fn=binary_accuracy, metric_name="acc")

    loss, acc = my.evaluate(trained_ae, valid_loader)
    print("\nTest: loss=%.4f, acc=%.4f" % (loss, acc))