from torch import nn
from torch.nn import functional as F
from torch.autograd import grad
import torch.optim as optim
import itertools

from model import sobolev_filter, stable_norm

SOBOLEV_C = 5.
SOBOLEV_S = 0.
EXPONENT = 2.

DUAL_EXPONENT = 1 / (1 - 1/EXPONENT) if EXPONENT != 1 else float('inf')


def d_loss(discriminator, x_true, x_gen):
	sobolev_true = sobolev_filter(x_true, c=SOBOLEV_C, s=SOBOLEV_S)
	lamb = stable_norm(sobolev_true, EXPONENT).mean()
	dual_sobolev_true = sobolev_filter(x_true, c=SOBOLEV_C, s=-SOBOLEV_S)
	gamma = stable_norm(dual_sobolev_true, DUAL_EXPONENT).mean()

	# regularizer
    epsilon = torch.rand(x_true.shape[0], device=x_true.device).view(-1, 1, 1, 1).expand_as(x_true)
    x_hat = epsilon * x_true + (1 - epsilon) * x_gen
    x_hat.requires_grad = True
    d_x_hat = discriminator(x_hat)
    [gradients] = grad(d_x_hat, x_hat, create_graph=True, retain_graph=True,
                       grad_output=torch.ones_like(d_x_hat, device=d_x_hat.device)) # some magic here
    dual_sobolev_gradients = sobolev_filter(gradients, c=SOBOLEV_C, s=-SOBOLEV_S)
    ddx = stable_norm(dual_sobolev_gradients, DUAL_EXPONENT)
    d_reg = ((ddx / gamma - 1) ** 2).mean()

    d_gen = discriminator(x_gen)
    d_true = discriminator(x_true)
    wasserstein = (d_gen.mean() - d_true.mean()) / gamma
    
    d_reg_mean = (d_true ** 2).mean()
    return -wasserstein + lamb * d_reg + 1e-5 * d_reg_mean


def train_iterations(num_iterations, generator, discriminator, 
	train_dataset, z_dim, disc_iters_per_gen_iters = 5, device=torch.device('cpu'), callback=None, callback_freq = 100):
	train_dataset = itertools.cycle(train_dataset)
	d_opt = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0., .9))
	g_opt = optim.Adam(generator.parameters(), lr=1e-4, betas=(0., .9))
	for i in range(num_iterations:
		generator.eval()
		for _ in range(disc_iters_per_gen_iters):
			x = next(train_dataset).to(device)
			z = torch.randn((x.shape[0], z_dim), device=device)
			x.requires_grad = True
			z.requires_grad = True

			discriminator.zero_grad()
			x_gen = generator(z)

			loss = d_loss(discriminator)
			loss.backward()
			d_opt.step()
		generator.train()
		generator.zero_grad()
		z = torch.randn((x.shape[0], z_dim), device=device)
		z.requires_grad = True
		x_gen = generator(z)
		loss = -discriminator(x_gen).mean()
		loss.backward()
		g_opt.step()
		if i % callback_freq == 0 and callback is not None:
			callback(generator, discriminator)
