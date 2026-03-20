import math

import torch


class GaussianRF(object):
    """Gaussian random field sampler adapted from the original FNO scripts."""

    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None, dtype=torch.float64):
        super().__init__()
        if boundary != "periodic":
            raise ValueError("Only periodic random fields are supported.")

        self.dim = dim
        self.size = size
        self.device = device
        self.dtype = dtype

        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - dim))

        freq = torch.fft.fftfreq(size, d=1.0 / size, device=device, dtype=dtype)
        if dim == 1:
            k2 = freq**2
        elif dim == 2:
            kx, ky = torch.meshgrid(freq, freq, indexing="ij")
            k2 = kx**2 + ky**2
        elif dim == 3:
            kx, ky, kz = torch.meshgrid(freq, freq, freq, indexing="ij")
            k2 = kx**2 + ky**2 + kz**2
        else:
            raise ValueError(f"Unsupported dimension: {dim}")

        scale = (size**dim) * math.sqrt(2.0) * sigma
        self.sqrt_eig = scale * ((4 * (math.pi**2) * k2 + tau**2) ** (-alpha / 2.0))
        self.sqrt_eig.reshape(-1)[0] = 0.0

    def sample(self, n_samples):
        coeff_real = torch.randn((n_samples,) + (self.size,) * self.dim, device=self.device, dtype=self.dtype)
        coeff_imag = torch.randn((n_samples,) + (self.size,) * self.dim, device=self.device, dtype=self.dtype)
        coeff = torch.complex(coeff_real, coeff_imag)
        coeff = coeff * self.sqrt_eig
        samples = torch.fft.ifftn(coeff, dim=tuple(range(1, self.dim + 1))).real
        return samples
