from os.path import basename, join

import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as T

from cellcount.utils import (ChunkSampler, ImageWithMask, train,
                             get_val_example, gpu_dtype, save_epoch_image)
from cellcount.models import FPN
from cellcount.losses import fpn_loss


BBBC = '/home/cxh/playground/bbbc/'

transform = T.Compose([T.Scale((256)), T.ToTensor()])

NUM_TRAIN = 1000
NUM_VAL = 200

train_data = ImageWithMask(join(BBBC, 'BBBC005_v1_ground_truth/'),
                           transform=transform)
train_data.imgs = [(join(BBBC, 'BBBC005_v1_images/jpg/%s') % basename(i), i)
                   for i, _ in train_data.imgs]

loader_train = DataLoader(train_data, batch_size=2,
                          sampler=ChunkSampler(NUM_TRAIN, 0))
loader_val = DataLoader(train_data, batch_size=2,
                        sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))


x_var, y_var = get_val_example()
_, _, h, w = x_var.size()

fpn = FPN(h, w).type(gpu_dtype)

lr = 5e-4
epochs = 1000
for i in range(epochs):
    print('epoch: %s' % i)
    if i > 0 and (i % 20 == 0):
        lr *= .95
    optimizer = optim.Adam(fpn.parameters(), lr=lr, weight_decay=0.)
    train(loader_train, fpn, fpn_loss, optimizer)
    if (i % 20 == 0):
        save_epoch_image(x_var, y_var, fpn, i)
