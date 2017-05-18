from os.path import basename

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import sampler

import torchvision.datasets as dset

import matplotlib.pyplot as plt
import numpy as np

gpu_dtype = torch.cuda.FloatTensor


def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


class Flatten(nn.Module):

    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        # "flatten" the C * H * W values into a single vector per image
        return x.view(N, -1)


class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """

    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class ImageWithMask(dset.ImageFolder):

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        target = self.loader(target)
        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)

        return img, target


class ImageWithCounts(dset.ImageFolder):

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        target = int(basename(path).split('_')[-4][1:])
        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_val_example(loader):
    for t, (x, y) in enumerate(loader):
        x_var = Variable(x.type(gpu_dtype))
        y_var = Variable(y.type(gpu_dtype))
        return x_var, y_var


def show(img, ax):
    ax.imshow(np.transpose(img, (1, 2, 0)), interpolation='nearest')


def make_grid(imgs, padding=20):
    N, C, H, W = imgs.size()

    M = np.zeros((C, H, W * N + (N - 1) * padding))
    for i in range(N):
        s = i * (W + padding)
        M[:, :, s:(s + W)] = imgs[i].cpu().data.numpy()
    if C == 1:
        M = np.repeat(M, 3, 0)
    return M


def save_epoch_image(x_var, y_var, model, epoch):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))

    show(make_grid(x_var), ax1)
    show(make_grid(y_var), ax2)
    show(make_grid(model(x_var)[-1], padding=5), ax3)

    fig.savefig('./test_images/FPN_epoch_%s.png' % epoch, dpi=300)


def train(loader_train, model, loss_fn, optimizer, num_epochs=1,
          print_every=10):
    for epoch in range(num_epochs):
        model.train()
        for t, (x, y) in enumerate(loader_train):
            x_var = Variable(x.type(gpu_dtype))
            y_var = Variable(y.type(gpu_dtype))

            scores = model(x_var)

            loss = loss_fn(scores, y_var)
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
