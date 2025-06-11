import diffusion.noise_lib as noise_lib
from types import SimpleNamespace
import torch

config = SimpleNamespace(
    noise=SimpleNamespace(
        type="loglinear",
        sigma_min=0.01,
        sigma_max=1.0
    )
)

sampling_eps = 1e-3
t = (1 - sampling_eps) * torch.rand(4) + sampling_eps

noise = noise_lib.get_noise(config)
sigma, dsigma = noise(t)
print(sigma[:, None, None].shape)