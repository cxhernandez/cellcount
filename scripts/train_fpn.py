from os.path import basename, join

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import visdom

from cellcount.utils import (ChunkSampler, ImageWithMask, train, test,
                             get_val_example, push_epoch_image,
                             save_checkpoint)
from cellcount.models import FPN
from cellcount.losses import fpn_loss

vis = visdom.Visdom(port=8080)


BBBC = '/home/cxh/playground/bbbc/'
NUM_TRAIN = 480
NUM_VAL = 120
BATCH_SIZE = 5
gpu_dtype = torch.cuda.FloatTensor

train_data = ImageWithMask(join(BBBC, 'BBBC005_v1_ground_truth/'))
train_data.imgs = [(join(BBBC, 'BBBC005_v1_images/jpg/%s') % basename(i), i)
                   for i, _ in train_data.imgs]

loader_train = DataLoader(train_data, batch_size=BATCH_SIZE,
                          sampler=ChunkSampler(NUM_TRAIN, 0))
loader_val = DataLoader(train_data, batch_size=BATCH_SIZE,
                        sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))


x_var, y_var = get_val_example(loader_val, gpu_dtype)
_, _, h, w = x_var.size()

fpn = FPN(h, w).type(gpu_dtype)

lr = 1e-3
epochs = 100
best_loss = 1E6
optimizer = optim.Adam(fpn.parameters(), lr=lr)
for epoch in range(epochs):
    print('epoch: %s' % epoch)

    if epoch > 0 and (epoch % 20 == 0):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= .5

    train(loader_train, fpn, fpn_loss, optimizer, gpu_dtype)
    val_loss = test(loader_val, fpn, fpn_loss, gpu_dtype)
    is_best = val_loss < best_loss
    if is_best:
        best_loss = val_loss
    save_checkpoint({
        'epoch': epoch,
        'fpn': fpn.state_dict(),
        'avg_val_loss': val_loss,
        'optimizer': optimizer.state_dict(),
    }, is_best)
    push_epoch_image(x_var, y_var, fpn, vis, epoch)
