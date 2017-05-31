from os.path import basename, join

import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as T

import visdom

from cellcount.utils import (ChunkSampler, ImageWithMask, train, test,
                             get_val_example, gpu_dtype, push_epoch_image,
                             save_checkpoint)
from cellcount.models import FPN
from cellcount.losses import fpn_loss

vis = visdom.Visdom(port=8080)


BBBC = '/home/cxh/playground/bbbc/'

transform = T.Compose([T.Scale((256)), T.ToTensor()])

NUM_TRAIN = 1000
NUM_VAL = 200
BATCH_SIZE = 2

train_data = ImageWithMask(join(BBBC, 'BBBC005_v1_ground_truth/'),
                           transform=transform)
train_data.imgs = [(join(BBBC, 'BBBC005_v1_images/jpg/%s') % basename(i), i)
                   for i, _ in train_data.imgs]

loader_train = DataLoader(train_data, batch_size=NUM_VAL,
                          sampler=ChunkSampler(NUM_TRAIN, 0))
loader_val = DataLoader(train_data, batch_size=NUM_VAL,
                        sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))


x_var, y_var = get_val_example()
_, _, h, w = x_var.size()

fpn = FPN(h, w).type(gpu_dtype)

lr = 5e-4
epochs = 1000
best_loss = 1E6
for epoch in range(epochs):
    print('epoch: %s' % epoch)
    if epoch > 0 and (epoch % 20 == 0):
        lr *= .95
    optimizer = optim.Adam(fpn.parameters(), lr=lr, weight_decay=0.)
    train(loader_train, fpn, fpn_loss, optimizer)
    val_loss = test(loader_val, fpn, fpn_loss)
    is_best = val_loss < best_loss
    save_checkpoint({
        'epoch': epoch,
        'fpn': fpn.state_dict(),
        'avg_val_loss': val_loss,
        'optimizer': optimizer.state_dict(),
    }, is_best)
    push_epoch_image(x_var, y_var, fpn, vis, epoch)
