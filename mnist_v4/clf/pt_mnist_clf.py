import sys
sys.path.insert(0, '..') ## '../..' for parent-parent directory

import torch
import torch.nn as nn

import os
import argparse
import numpy as np
import random

from mnist import load_mnist, load_fashion_mnist
import pt_trainer as my


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


def get_mlp_model(n_classes):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 256),
        nn.ReLU(),
        nn.LayerNorm(256),
        nn.Linear(256, n_classes),)
    return model


def get_cnn_model(n_classes):
    model = nn.Sequential(
        nn.Conv2d(1, 32, (3, 3), stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, (3, 3), stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Flatten(),
        nn.Linear(64*7*7, 256),
        nn.ReLU(),
        nn.Linear(256, n_classes),)
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
    p.add_argument("--log_dir", type=str, default="log_pt_mnist_clf10")
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
    def accuracy(y_pred, y_true):
        return torch.eq(y_pred.argmax(-1), y_true).float().mean()
    
    model = get_mlp_model(n_classes=10) if args.mlp else get_cnn_model(n_classes=10)
    model = model.to(device)
    args.log_dir += "_mnp" if args.mlp else "_cnn"

    clf = my.TrainerWithMetric(model)
    clf.compile(optim=torch.optim.Adam(model.parameters(), lr=1e-4),
                loss_fn=nn.CrossEntropyLoss(),
                metric_fn=accuracy, metric_name="acc")

    hist = my.train_with_metric(clf, train_loader, valid_loader, args)
    my.plot_progress(hist, args)
    
    ## Evaluation:
    model_weights = os.path.join(args.log_dir, args.log_dir + "_weights.pth")
    trained_model = get_mlp_model(n_classes=10) if args.mlp else get_cnn_model(n_classes=10)
    trained_mode = trained_model.to(device)
    trained_model.load_state_dict(torch.load(model_weights))

    trained_clf = my.TrainerWithMetric(model)
    trained_clf.compile(optim=torch.optim.Adam(trained_clf.parameters(), lr=1e-4),
                        loss_fn=nn.CrossEntropyLoss(),
                        metric_fn=accuracy, metric_name="acc")

    loss, acc = my.evaluate(trained_clf, valid_loader)
    print("\nTest: loss=%.4f, acc=%.4f" % (loss, acc))