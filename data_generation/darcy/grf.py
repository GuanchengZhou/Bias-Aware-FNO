"""Gaussian random field helpers for Darcy coefficient generation."""

from __future__ import annotations

import numpy as np
from scipy.fft import idctn


def sample_grf(
    rng: np.random.Generator,
    resolution: int,
    alpha: float = 2.0,
    tau: float = 3.0,
) -> np.ndarray:
    """Sample a 2D GRF using the FNO-Darcy DCT construction."""

    xi = rng.standard_normal((resolution, resolution))
    kx, ky = np.meshgrid(np.arange(resolution), np.arange(resolution), indexing="ij")
    coef = (tau ** (alpha - 1.0)) * ((np.pi**2 * (kx**2 + ky**2) + tau**2) ** (-alpha / 2.0))
    kl = resolution * coef * xi
    kl[0, 0] = 0.0
    return idctn(kl, type=2, norm="ortho").astype(np.float64)


def sample_threshold_coefficients(
    rng: np.random.Generator,
    resolution: int,
    alpha: float = 2.0,
    tau: float = 3.0,
    coeff_low: float = 4.0,
    coeff_high: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the thresholded Darcy coefficient field and the latent GRF."""

    latent = sample_grf(rng=rng, resolution=resolution, alpha=alpha, tau=tau)
    coeff = np.where(latent >= 0.0, coeff_high, coeff_low)
    return coeff.astype(np.float64), latent.astype(np.float64)
