import torch
import torch.nn as nn

gpu_dtype = torch.cuda.FloatTensor


def tv_loss(img, tv_weight=1E-6):
    """
    Compute total variation loss.

    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # Your implementation should be vectorized and not require any loops!
    N, C, H, W = img.size()
    f = img[:, :, :-1, :-1]
    g = img[:, :, :-1, 1:]
    h = img[:, :, 1:, :-1]
    return tv_weight * torch.sum((f - g)**2. + (f - h)**2.)


def bloss(x, lv, y, eps=1E-6):
    t = nn.Threshold(-6., 0.)
    return torch.mean((torch.abs(y - x)) / (2. * torch.exp(t(lv))) + t(lv))


def fpn_loss(x, y):

    loss = 0.
    for x_i, v_i in zip(*x):
        n, _, h, w = x_i.size()
        p_i = nn.AdaptiveAvgPool2d(output_size=(h, w))
        y_i = torch.sum(p_i(y), 1)
        loss += bloss(x_i, v_i, y_i)
        loss += tv_loss(x_i)
    return loss


def counter_loss(x, y):
    x, lv = x
    return bloss(x, lv, y)
