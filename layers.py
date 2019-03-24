import torch 
from torch import nn
from torch.nn import functional as F

class AvgUnpooling2d(nn.Module):
    def __init__(self, filters):
        if type(filters) == int:
            self.filters = (filters, filters)
        else:
            filters = tuple(filters)
            assert len(filters) == 2
            self.filters = filters

    def forward(self, x):
        # n c h w
        n, c, h, w = x.shape
        fh, fw = self.filters

        out = x.view(n, c, h, 1, w, 1).expand(n, c, h, fh, w, hw).contiguous().view(n, c, fh * h, fw * w )
        return out


def xavier_init_conv_(conv):
    nn.init.xavier_uniform_(conv.weight)
    nn.init.xavier_uniform_(conv.bias)

class ResBlock(nn.Module):
    def __init__(self, filters, resample=None, normalize=False):
        layers = []
        if self.normalize:
            layers.append(nn.BatchNorm2d(filters))

        layers.append(nn.ReLU())
        if resample == 'up':
            layers.append(AvgUnpooling2d(2))
        layers.append(nn.ReplicationPad2d(1))
        layers.append(nn.Conv2d(filters, filters, 3))
        if self.normalize:
            layers.append(nn.BatchNorm2d(filters))
        layers.append(nn.ReLU())
        layers.append(nn.ReplicationPad2d(1))
        layers.append(nn.Conv2d(filters, filters, 3))
        if resample == 'down':
            layers.append(nn.AvgPool2d(2))
        self.update = nn.Sequential(*layers)

        if resample == 'down':
            self.shortcut = nn.Sequential(
                nn.Conv2d(filters, filters, 1),
                nn.AvgPool2d())
            xavier_init_conv_(self.shortcut[0])
        elif resample == 'up':
            self.shortcut = nn.Sequential(
                AvgUnpooling2d(2),
                nn.Conv2d(filters, filters, 1))
            xavier_init_conv_(self.shortcut[1])
        elif resample is None:
            self.shortcut = lambda x: x
        else:
            raise ValueError('wrong resample')

    def forward(self, x):
        update = self.update(x)
        skip = self.shortcut(x)
        return skip + update

class ResBlockOptimized(nn.Module):
    def __init__(self, filters, c_in=None):
        if c_in is None:
            c_in = filters
        self.update = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(c_in, filters, 3),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(filters, filters, 3),
            nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(c_in, filters, 1))
        xavier_init_conv_(self.shortcut[1])

    def forward(self, x):
        update = self.update(x)
        skip = self.shortcut(x)
        return skip + update
