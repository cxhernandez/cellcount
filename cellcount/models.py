import torch.nn as nn


def ConvBNReLUPool(i, o, kernel_size=(3, 3), padding=0, p=0.5, pool=False):
    model = [nn.Conv2d(i, o, kernel_size=kernel_size, padding=padding),
             nn.BatchNorm2d(o),
             nn.ReLU(inplace=True)
             ]
    if pool:
        model += [nn.MaxPool2d(2, ceil_mode=True)]
    if p > 0.:
        model += [nn.Dropout2d(p)]
    return nn.Sequential(*model)


class FPN(nn.Module):

    def __init__(self, height, width, h=4, ratio=2, d=128):
        """
        Initialize Feature Pyramid Network (FPN)
        """
        super(FPN, self).__init__()

        self.pyramid = {}
        self.conv_1 = ConvBNReLUPool(3, 1, padding=1, p=0.1)
        self.h = h
        self.d = d
        self.conv_2 = []

        heights, widths = [], []
        for i in range(self.h):
            height, width = height // ratio, width // ratio
            heights.append(height)
            widths.append(width)
            self.pyramid['down%s' % i] = nn.AdaptiveAvgPool2d(
                output_size=(height, width)).cuda()

        for i in range(self.h):
            self.pyramid['across%s' % i] = ConvBNReLUPool(
                1, self.d, p=0.1, padding=1, kernel_size=(3, 3)).cuda()
            height, width = heights[-(i + 1)], widths[-(i + 1)]
            self.pyramid['up%s' % i] = nn.UpsamplingBilinear2d(
                size=(height, width)).cuda()

        for i in range(self.h):
            self.conv_2.append(ConvBNReLUPool(
                self.d, 1, padding=1, kernel_size=(3, 3), p=0.1).cuda())

    def forward(self, x):
        """
        Foward Pass through FPN
        """
        y_1 = self.conv_1(x)

        down_sampled = [y_1]
        for i in range(self.h):
            down_sampled.append(self.pyramid['down%s' % i](down_sampled[-1]))

        up_sampled = [down_sampled[-1].repeat(1, self.d, 1, 1)]
        for i in range(self.h - 1):
            up_2 = self.pyramid['across%s' % i](down_sampled[-(i + 1)])
            up_1 = self.pyramid['up%s' % i](up_sampled[-1])
            up_sampled.append(up_1 + up_2)

        for i, item in enumerate(up_sampled):
            up_sampled[i] = self.conv_2[i](item)

        return up_sampled
