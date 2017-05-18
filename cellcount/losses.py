import torch
import torch.nn as nn

gpu_dtype = torch.cuda.FloatTensor


def fpn_loss(x, y):
    loss_fn = nn.SmoothL1Loss().type(gpu_dtype)
    loss = 0.
    for x_i in x:
        _, _, h, w = x_i.size()
        p_i = nn.AdaptiveAvgPool2d(output_size=(h, w)).cuda()
        y_i = torch.sum(p_i(y), 1)
        loss += torch.sum(loss_fn(x_i, y_i))
    return loss
