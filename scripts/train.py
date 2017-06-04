from os.path import join
from collections import OrderedDict

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

import visdom

from cellcount.utils import (ChunkSampler, ImageWithCount, train, test,
                             get_val_example, push_epoch_image_count,
                             save_checkpoint, reset)
from cellcount.models import FPN, Counter
from cellcount.losses import counter_loss

vis = visdom.Visdom(port=8080)


BBBC = '/home/cxh/playground/bbbc/'
NUM_TRAIN = 4000
NUM_VAL = 1000
BATCH_SIZE = 5
gpu_dtype = torch.cuda.FloatTensor
transform = T.Compose([T.Scale((256)), T.RandomHorizontalFlip(), T.ToTensor()])

train_data = ImageWithCount(join(BBBC, 'BBBC005_v1_images/'),
                            transform=transform)

loader_train = DataLoader(train_data, batch_size=BATCH_SIZE,
                          sampler=ChunkSampler(NUM_TRAIN, 0))
loader_val = DataLoader(train_data, batch_size=BATCH_SIZE,
                        sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))


x_var, y_var = get_val_example(loader_val, gpu_dtype)
_, _, h, w = x_var.size()

fpn = FPN(h, w).type(gpu_dtype)
reset(fpn)
checkpoint = torch.load('fpn_best.pth.tar')
fpn.load_state_dict(checkpoint['fpn'])


count = Counter(h // 2, w // 2).type(gpu_dtype)

model = nn.Sequential(OrderedDict([('fpn', fpn), ('counter', count)]))

lr = 1e-4
epochs = 100
loss_fn = counter_loss
val_loss_fn = nn.MSELoss()
best_loss = 1E6
optimizer = optim.Adam(model.counter.parameters(), lr=lr)
for epoch in range(epochs):
    print('epoch: %s' % epoch)

    if epoch > 0 and (epoch % 20 == 0):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= .5

    train(loader_train, model, loss_fn, optimizer, gpu_dtype)
    val_loss = test(loader_val, model, val_loss_fn, gpu_dtype)
    is_best = val_loss < best_loss
    if is_best:
        best_loss = val_loss
    save_checkpoint({
        'epoch': epoch,
        'model': model.state_dict(),
        'avg_val_loss': val_loss,
        'optimizer': optimizer.state_dict(),
    }, is_best)
    push_epoch_image_count(x_var, y_var, model, vis, epoch)
