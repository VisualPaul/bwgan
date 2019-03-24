import torch

from torch import nn
from torch.nn import functional as F
from torch.autograd import grad

from layers import ResBlock, ResBlockOptimized, xavier_init_conv_

class GeneratorMNIST(nn.Module):
    def __init__(self, hidden_size, filters=64):
        self.filters = filters
        self.pre_process = nn.Linear(hidden_size, filters * 7 * 7)  # 2 res blocks - starting with 7 x 7
        self.main = nn.Sequential(
            ResBlock(filters, resample='up', normalize=True),
            ResBlock(filters, resample='up', normalize=True),
        )
        self.post_process = nn.Sequential(
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            layers.append(nn.ReplicationPad2d(1)),
            layers.append(nn.Conv2d(filters, 1, 3)))
        xavier_init_conv_(self.post_process[3])

    def forward(self, x):
        x = self.pre_process(x).view(-1, self.filters, 7, 7)
        x = self.main(x)
        return torch.tanh(self.post_process(x))


class DiscriminatorMNIST(nn.Module):
    def __init__(self, hidden_size, filters=64):
        self.pre_process = ResBlockOptimized(filters, 1)
        self.main = nn.Sequential(
            ResBlock(filters, resample='down'),
            ResBlock(filters))
        self.post_process = nn.Linear(filters, 1)

    def forward(self, x):
        x = self.pre_process(x)
        x = self.main(x)
        x = F.relu(x)
        x = x.mean(dim=3).mean(dim=2)
        x = self.post_process(x)
        return x

class Generator(nn.Module):
    def __init__(self, hidden_size, filters=128):
        self.filters = filters
        self.pre_process = nn.Linear(hidden_size, filters * 4 * 4)  # 2 res blocks - starting with 7 x 7
        self.main = nn.Sequential(
            ResBlock(filters, resample='up', normalize=True),
            ResBlock(filters, resample='up', normalize=True),
            ResBlock(filters, resample='up', normalize=True),
        )
        self.post_process = nn.Sequential(
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            layers.append(nn.ReplicationPad2d(1)),
            layers.append(nn.Conv2d(filters, 1, 3)))
        xavier_init_conv_(self.post_process[3])

    def forward(self, x):
        x = self.pre_process(x).view(-1, self.filters, 4, 4)
        x = self.main(x)
        return torch.tanh(self.post_process(x))


class Discriminator(nn.Module):
    def __init__(self, hidden_size, filters=128):
        self.pre_process = ResBlockOptimized(filters, 1)
        self.main = nn.Sequential(
            ResBlock(filters, resample='down'),
            ResBlock(filters),
            ResBlock(filters))
        self.post_process = nn.Linear(filters, 1)

    def forward(self, x):
        x = self.pre_process(x)
        x = self.main(x)
        x = F.relu(x)
        x = x.mean(dim=3).mean(dim=2)
        x = self.post_process(x)
        return x

def sobolev_filter(x, c, s):
    # x: N C H W
    imag = torch.zeros(x.shape, dtype=torch.float64)
    real = x.to(torch.float64)
    fft_x = torch.fft(torch.stack([real, imag], dim=-1), 2, normalized=False)
    #print(fft_x)
    sx, sy = fft_x.shape[2:4]
    x = torch.arange(sx)
    x = torch.min(x, sx-x)
    x = x.to(torch.float64) / float(sx // 2)
    y = torch.arange(sy)
    y = torch.min(y, sy-y)
    y = y.to(torch.float64) / float(sy // 2)
    X = x.view(1, 1, -1,  1).expand(1, 1, sx, sy)
    Y = x.view(1, 1,  1, -1).expand(1, 1, sx, sy)

    scale = (1 + c * (X ** 2 + Y ** 2)) ** (s / 2)
    scale = scale[..., None].expand(-1, -1, -1, -1, 2)

    fft_x = scale * fft_x
    result = torch.ifft(fft_x, 2, normalized=False)[...,0] # real part
    return result.to(torch.float32)

def stable_norm(x, order):
    x = x.view(x.shape[0], -1)
    alpha = (torch.abs(x) + 1e-5).max(dim=1)
    return alpha torch.norm(x/alpha[:,None], p=order, dim=1)

#sobolev_true = sobolev_filter(x_true, c=SOBOLEV_C, s=SOBOLEV_S)

def regularizer(discriminator, x, x_gen):
    epsilon = torch.rand(x.shape[0], device=x.device).view(-1, 1, 1, 1).expand_as(x)
    x_hat = epsilon * x + (1 - epsilon) * x_gen
    x_hat.requires_grad = True
    d_x_hat = discriminator(x_hat)
    [gradients] = grad(d_x_hat, x_hat, create_graph=True, retain_graph=True,
                       grad_output=torch.ones_like(d_x_hat, device=d_x_hat.device)) # some magic here
    dual_sobolev_gradients = sobolev_filter(gradients, c=SOBOLEV_C, s=SOBOLEV_S)



if __name__ == '__main__':
    import scipy
    import scipy.misc
    import PIL
    import numpy as np

    from sys import platform as sys_pf
    if sys_pf == 'darwin':
        import matplotlib
        matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt


    img = scipy.misc.face()
    print(img.shape)

    img_tensor = torch.from_numpy(np.array(PIL.Image.fromarray(img).resize((32, 32), PIL.Image.BILINEAR)) / 256.)
    img_tensor = img_tensor.permute(2, 0, 1)[None, ...]
    result = sobolev_filter(img_tensor, c=5.0, s=1)

    plt.figure('input')
    plt.imshow(img_tensor[0, 0, ...].numpy())
    plt.colorbar()

    plt.figure('result')
    result_show = result[0]
    plt.imshow(result_show[0, ...].numpy())
    plt.colorbar()

    print(result)

    plt.figure('diff')
    diff = img_tensor[0] - result[0]
    plt.imshow(diff[0, ...].numpy())
    plt.colorbar()
    plt.show()
