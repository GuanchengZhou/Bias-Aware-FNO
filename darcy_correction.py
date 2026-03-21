#!/usr/bin/env python3
"""Structured flux correction model for Darcy-FNO."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from fourier_2d_darcy_fem import FNO2d, SpectralConv2d


def make_grid(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.linspace(0.0, 1.0, size, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)


def enforce_zero_boundary(field: torch.Tensor) -> torch.Tensor:
    out = field.clone()
    out[:, 0, :] = 0.0
    out[:, -1, :] = 0.0
    out[:, :, 0] = 0.0
    out[:, :, -1] = 0.0
    return out


def interior_mask(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros((1, size, size), device=device, dtype=dtype)
    if size > 2:
        mask[:, 1:-1, 1:-1] = 1.0
    return mask


def x_face_mask(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros((1, size - 1, size), device=device, dtype=dtype)
    if size > 2:
        mask[:, :, 1:-1] = 1.0
    return mask


def y_face_mask(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros((1, size, size - 1), device=device, dtype=dtype)
    if size > 2:
        mask[:, 1:-1, :] = 1.0
    return mask


def boundary_distance(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.linspace(0.0, 1.0, size, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
    return torch.minimum(torch.minimum(grid_x, 1.0 - grid_x), torch.minimum(grid_y, 1.0 - grid_y))


def boundary_normals(size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.linspace(0.0, 1.0, size, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
    distances = torch.stack([grid_x, 1.0 - grid_x, grid_y, 1.0 - grid_y], dim=0)
    nearest = torch.argmin(distances, dim=0)

    nx = torch.zeros((size, size), device=device, dtype=dtype)
    ny = torch.zeros((size, size), device=device, dtype=dtype)
    nx = torch.where(nearest == 0, torch.full_like(nx, -1.0), nx)
    nx = torch.where(nearest == 1, torch.full_like(nx, 1.0), nx)
    ny = torch.where(nearest == 2, torch.full_like(ny, -1.0), ny)
    ny = torch.where(nearest == 3, torch.full_like(ny, 1.0), ny)
    return nx, ny


def forward_diff_x(field: torch.Tensor, h: float) -> torch.Tensor:
    return (field[:, 1:, :] - field[:, :-1, :]) / h


def forward_diff_y(field: torch.Tensor, h: float) -> torch.Tensor:
    return (field[:, :, 1:] - field[:, :, :-1]) / h


def centered_gradient(field: torch.Tensor, h: float) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x = torch.zeros_like(field)
    grad_y = torch.zeros_like(field)

    grad_x[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2.0 * h)
    grad_x[:, 0, :] = (field[:, 1, :] - field[:, 0, :]) / h
    grad_x[:, -1, :] = (field[:, -1, :] - field[:, -2, :]) / h

    grad_y[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2.0 * h)
    grad_y[:, :, 0] = (field[:, :, 1] - field[:, :, 0]) / h
    grad_y[:, :, -1] = (field[:, :, -1] - field[:, :, -2]) / h
    return grad_x, grad_y


def cell_to_face_x(field: torch.Tensor) -> torch.Tensor:
    return 0.5 * (field[:, 1:, :] + field[:, :-1, :])


def cell_to_face_y(field: torch.Tensor) -> torch.Tensor:
    return 0.5 * (field[:, :, 1:] + field[:, :, :-1])


def harmonic_face_coefficients(coeff: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    coeff_x = 2.0 * coeff[:, 1:, :] * coeff[:, :-1, :] / (coeff[:, 1:, :] + coeff[:, :-1, :] + eps)
    coeff_y = 2.0 * coeff[:, :, 1:] * coeff[:, :, :-1] / (coeff[:, :, 1:] + coeff[:, :, :-1] + eps)
    return coeff_x, coeff_y


def divergence(qx: torch.Tensor, qy: torch.Tensor, h: float) -> torch.Tensor:
    batch_size, nx_minus_one, ny = qx.shape
    nx = nx_minus_one + 1
    out = torch.zeros((batch_size, nx, ny), device=qx.device, dtype=qx.dtype)
    if nx > 2 and ny > 2:
        out[:, 1:-1, 1:-1] = (
            (qx[:, 1:, 1:-1] - qx[:, :-1, 1:-1]) / h
            + (qy[:, 1:-1, 1:] - qy[:, 1:-1, :-1]) / h
        )
    return out


def laplacian(field: torch.Tensor, h: float) -> torch.Tensor:
    return divergence(forward_diff_x(field, h), forward_diff_y(field, h), h)


def darcy_operator(coeff: torch.Tensor, field: torch.Tensor, h: float) -> torch.Tensor:
    coeff_x, coeff_y = harmonic_face_coefficients(coeff)
    grad_x = forward_diff_x(field, h)
    grad_y = forward_diff_y(field, h)
    return -divergence(coeff_x * grad_x, coeff_y * grad_y, h)


def darcy_flux(coeff: torch.Tensor, field: torch.Tensor, h: float) -> tuple[torch.Tensor, torch.Tensor]:
    coeff_x, coeff_y = harmonic_face_coefficients(coeff)
    grad_x = forward_diff_x(field, h)
    grad_y = forward_diff_y(field, h)
    return -(coeff_x * grad_x), -(coeff_y * grad_y)


def face_to_cell_magnitude(qx: torch.Tensor, qy: torch.Tensor) -> torch.Tensor:
    batch_size, nx_minus_one, ny = qx.shape
    nx = nx_minus_one + 1
    out = torch.zeros((batch_size, nx, ny), device=qx.device, dtype=qx.dtype)
    qx_left = torch.zeros_like(out)
    qx_right = torch.zeros_like(out)
    qy_down = torch.zeros_like(out)
    qy_up = torch.zeros_like(out)
    qx_left[:, 1:, :] = qx
    qx_right[:, :-1, :] = qx
    qy_down[:, :, 1:] = qy
    qy_up[:, :, :-1] = qy
    avg_x = 0.5 * (qx_left + qx_right)
    avg_y = 0.5 * (qy_down + qy_up)
    return torch.sqrt(avg_x.square() + avg_y.square() + 1e-12)


def smooth_face_norm(qx: torch.Tensor, qy: torch.Tensor) -> torch.Tensor:
    value = qx.square().mean(dim=(1, 2)) + qy.square().mean(dim=(1, 2))
    if qx.shape[1] > 1:
        value = value + (qx[:, 1:, :] - qx[:, :-1, :]).square().mean(dim=(1, 2))
    if qx.shape[2] > 1:
        value = value + (qx[:, :, 1:] - qx[:, :, :-1]).square().mean(dim=(1, 2))
    if qy.shape[1] > 1:
        value = value + (qy[:, 1:, :] - qy[:, :-1, :]).square().mean(dim=(1, 2))
    if qy.shape[2] > 1:
        value = value + (qy[:, :, 1:] - qy[:, :, :-1]).square().mean(dim=(1, 2))
    return value


def masked_mean_square(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(mask.sum(dim=(1, 2)), min=1.0)
    return (field.square() * mask).sum(dim=(1, 2)) / denom


def masked_mean(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(mask.sum(dim=(1, 2)), min=1.0)
    return (field * mask).sum(dim=(1, 2)) / denom


def masked_face_mean_square(qx: torch.Tensor, qy: torch.Tensor, mask_x: torch.Tensor, mask_y: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(mask_x.sum(dim=(1, 2)) + mask_y.sum(dim=(1, 2)), min=1.0)
    numer = (qx.square() * mask_x).sum(dim=(1, 2)) + (qy.square() * mask_y).sum(dim=(1, 2))
    return numer / denom


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff_norm = torch.norm((pred - target).reshape(pred.shape[0], -1), dim=1)
    target_norm = torch.norm(target.reshape(target.shape[0], -1), dim=1)
    return diff_norm / torch.clamp(target_norm, min=1e-12)


def batch_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x.reshape(x.shape[0], -1) * y.reshape(y.shape[0], -1)).sum(dim=1)


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.exp(logvar) + mu.square() - 1.0 - logvar).mean(dim=tuple(range(1, mu.ndim)))


def latent_channels_for_ablation(ablation: str) -> int:
    if ablation == "none":
        return 4
    if ablation == "direct-bias":
        return 1
    if ablation == "direct-flux":
        return 2
    raise ValueError(f"Unsupported ablation: {ablation}")


@dataclass
class LossBundle:
    total: torch.Tensor
    backbone_coarse: torch.Tensor
    state_fine: torch.Tensor
    pde: torch.Tensor
    flux: torch.Tensor
    reg: torch.Tensor
    mask: torch.Tensor
    nll: torch.Tensor
    kl_beta: torch.Tensor
    kl_var: torch.Tensor


class FlexibleFNO2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, width: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.padding = 9

        self.fc0 = nn.Linear(in_channels + 2, width)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = make_grid(x.shape[1], x.device, x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, grid], dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


class BackboneWrapper(nn.Module):
    def __init__(self, modes: int, width: int):
        super().__init__()
        self.model = FNO2d(modes, modes, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def init_small_output(linear: nn.Linear, *, std: float = 1e-3, bias: float = 0.0) -> None:
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    nn.init.constant_(linear.bias, bias)


class StructuredCorrectionNet(nn.Module):
    def __init__(self, modes: int, width: int):
        super().__init__()
        self.model = FlexibleFNO2d(in_channels=7, out_channels=4, modes1=modes, modes2=modes, width=width)
        # Avoid the tau=0 -> rhs=0 -> zero-gradient fixed point at initialization.
        init_small_output(self.model.fc2, std=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.1 * torch.tanh(self.model(x))


class DeterministicLatentHead(nn.Module):
    def __init__(self, modes: int, width: int, out_channels: int, init_bias: float = 0.0):
        super().__init__()
        self.model = FlexibleFNO2d(in_channels=7, out_channels=out_channels, modes1=modes, modes2=modes, width=width)
        init_small_output(self.model.fc2, std=1e-3, bias=init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.1 * torch.tanh(self.model(x))


class BayesianLatentHead(nn.Module):
    def __init__(self, modes: int, width: int, latent_channels: int):
        super().__init__()
        self.mu_head = FlexibleFNO2d(in_channels=7, out_channels=latent_channels, modes1=modes, modes2=modes, width=width)
        self.logvar_head = FlexibleFNO2d(
            in_channels=7,
            out_channels=latent_channels,
            modes1=modes,
            modes2=modes,
            width=width,
        )
        nn.init.zeros_(self.mu_head.fc2.weight)
        nn.init.zeros_(self.mu_head.fc2.bias)
        nn.init.zeros_(self.logvar_head.fc2.weight)
        nn.init.constant_(self.logvar_head.fc2.bias, -8.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = 0.1 * torch.tanh(self.mu_head(x))
        logvar = torch.clamp(self.logvar_head(x), min=-10.0, max=4.0)
        return mu, logvar


class UncertaintyHead(nn.Module):
    def __init__(self, modes: int, width: int):
        super().__init__()
        self.model = FlexibleFNO2d(in_channels=7, out_channels=1, modes1=modes, modes2=modes, width=width)
        nn.init.zeros_(self.model.fc2.weight)
        nn.init.constant_(self.model.fc2.bias, -6.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.model(x).squeeze(-1), min=-10.0, max=4.0)


class StructuredFluxCorrectionLayer(nn.Module):
    def __init__(
        self,
        *,
        cg_max_iter: int,
        cg_tol: float,
        interface_gamma: float = 10.0,
        interface_threshold: float = 0.5,
        boundary_delta: float = 0.15,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.cg_max_iter = int(cg_max_iter)
        self.cg_tol = float(cg_tol)
        self.interface_gamma = float(interface_gamma)
        self.interface_threshold = float(interface_threshold)
        self.boundary_delta = float(boundary_delta)
        self.eps = float(eps)

    def _grid_helpers(self, coeff: torch.Tensor) -> dict[str, torch.Tensor]:
        size = coeff.shape[-1]
        device = coeff.device
        dtype = coeff.dtype
        boundary_nx, boundary_ny = boundary_normals(size, device, dtype)
        return {
            "interior_mask": interior_mask(size, device, dtype),
            "x_face_mask": x_face_mask(size, device, dtype),
            "y_face_mask": y_face_mask(size, device, dtype),
            "d_boundary": boundary_distance(size, device, dtype).unsqueeze(0),
            "boundary_nx": boundary_nx.unsqueeze(0),
            "boundary_ny": boundary_ny.unsqueeze(0),
        }

    def _apply_dirichlet_operator(self, coeff: torch.Tensor, field: torch.Tensor, h: float) -> torch.Tensor:
        out = darcy_operator(coeff, field, h)
        out[:, 0, :] = field[:, 0, :]
        out[:, -1, :] = field[:, -1, :]
        out[:, :, 0] = field[:, :, 0]
        out[:, :, -1] = field[:, :, -1]
        return out

    def _cg_solve(self, coeff: torch.Tensor, rhs: torch.Tensor, h: float) -> torch.Tensor:
        rhs = enforce_zero_boundary(rhs)
        solution = torch.zeros_like(rhs)
        residual = rhs - self._apply_dirichlet_operator(coeff, solution, h)
        direction = residual.clone()
        residual_norm = batch_dot(residual, residual)

        for _ in range(self.cg_max_iter):
            ap = self._apply_dirichlet_operator(coeff, direction, h)
            alpha = residual_norm / torch.clamp(batch_dot(direction, ap), min=self.eps)
            alpha = alpha.view(-1, 1, 1)
            solution = enforce_zero_boundary(solution + alpha * direction)
            residual = residual - alpha * ap
            new_norm = batch_dot(residual, residual)
            if torch.sqrt(torch.max(new_norm)).item() <= self.cg_tol:
                break
            beta = (new_norm / torch.clamp(residual_norm, min=self.eps)).view(-1, 1, 1)
            direction = residual + beta * direction
            residual_norm = new_norm

        return enforce_zero_boundary(solution)

    def _finalize_outputs(
        self,
        coeff: torch.Tensor,
        u_backbone: torch.Tensor,
        tau_x: torch.Tensor,
        tau_y: torch.Tensor,
        helpers: dict[str, torch.Tensor],
        *,
        tau_bulk_x: torch.Tensor,
        tau_bulk_y: torch.Tensor,
        tau_int_x: torch.Tensor,
        tau_int_y: torch.Tensor,
        tau_bdry_x: torch.Tensor,
        tau_bdry_y: torch.Tensor,
        rho_int_x: torch.Tensor,
        rho_int_y: torch.Tensor,
        rho_bdry_x: torch.Tensor,
        rho_bdry_y: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h = 1.0 / float(coeff.shape[-1] - 1)
        rhs = divergence(tau_x, tau_y, h)
        correction = self._cg_solve(coeff, rhs, h)
        u_corrected = enforce_zero_boundary(u_backbone + correction)

        q_backbone_x, q_backbone_y = darcy_flux(coeff, u_backbone, h)
        q_corrected_x, q_corrected_y = darcy_flux(coeff, u_corrected, h)
        residual_backbone = 1.0 - darcy_operator(coeff, u_backbone, h)
        residual_corrected = darcy_operator(coeff, u_corrected, h) - 1.0

        return {
            "u_corrected": u_corrected,
            "b_h": correction,
            "tau_x": tau_x,
            "tau_y": tau_y,
            "tau_bulk_x": tau_bulk_x,
            "tau_bulk_y": tau_bulk_y,
            "tau_int_x": tau_int_x,
            "tau_int_y": tau_int_y,
            "tau_bdry_x": tau_bdry_x,
            "tau_bdry_y": tau_bdry_y,
            "rho_int_x": rho_int_x,
            "rho_int_y": rho_int_y,
            "rho_bdry_x": rho_bdry_x,
            "rho_bdry_y": rho_bdry_y,
            "residual_backbone": residual_backbone,
            "residual_corrected": residual_corrected,
            "flux_backbone_x": q_backbone_x,
            "flux_backbone_y": q_backbone_y,
            "flux_corrected_x": q_corrected_x,
            "flux_corrected_y": q_corrected_y,
            "interior_mask": helpers["interior_mask"],
            "x_face_mask": helpers["x_face_mask"],
            "y_face_mask": helpers["y_face_mask"],
            "d_boundary": helpers["d_boundary"],
        }

    def from_beta(
        self,
        coeff: torch.Tensor,
        u_backbone: torch.Tensor,
        beta: torch.Tensor,
        *,
        disable_interface_correction: bool = False,
        disable_boundary_correction: bool = False,
    ) -> dict[str, torch.Tensor]:
        size = coeff.shape[-1]
        h = 1.0 / float(size - 1)
        helpers = self._grid_helpers(coeff)

        beta_bulk = beta[..., 0]
        beta_n = beta[..., 1]
        beta_t = beta[..., 2]
        beta_b = beta[..., 3]

        lap = laplacian(u_backbone, h)
        grad_u_x = forward_diff_x(u_backbone, h)
        grad_u_y = forward_diff_y(u_backbone, h)
        coeff_x, coeff_y = harmonic_face_coefficients(coeff, eps=self.eps)
        lap_grad_x = forward_diff_x(lap, h)
        lap_grad_y = forward_diff_y(lap, h)

        log_coeff = torch.log(torch.clamp(coeff, min=self.eps))
        grad_log_coeff_x, grad_log_coeff_y = centered_gradient(log_coeff, h)
        eta = torch.sqrt(grad_log_coeff_x.square() + grad_log_coeff_y.square() + self.eps)
        rho_int = torch.sigmoid(self.interface_gamma * (eta - self.interface_threshold))
        rho_bdry = torch.exp(-(helpers["d_boundary"].square()) / (self.boundary_delta**2))

        coeff_grad_x, coeff_grad_y = centered_gradient(coeff, h)
        grad_norm = torch.sqrt(coeff_grad_x.square() + coeff_grad_y.square() + self.eps)
        normal_x = coeff_grad_x / grad_norm
        normal_y = coeff_grad_y / grad_norm

        beta_bulk_x = cell_to_face_x(beta_bulk)
        beta_bulk_y = cell_to_face_y(beta_bulk)
        beta_n_x = cell_to_face_x(beta_n)
        beta_n_y = cell_to_face_y(beta_n)
        beta_t_x = cell_to_face_x(beta_t)
        beta_t_y = cell_to_face_y(beta_t)
        beta_b_x = cell_to_face_x(beta_b)
        beta_b_y = cell_to_face_y(beta_b)

        rho_int_x = cell_to_face_x(rho_int)
        rho_int_y = cell_to_face_y(rho_int)
        rho_bdry_x = cell_to_face_x(rho_bdry)
        rho_bdry_y = cell_to_face_y(rho_bdry)

        abs_normal_x = torch.abs(cell_to_face_x(normal_x))
        abs_normal_y = torch.abs(cell_to_face_y(normal_y))
        abs_boundary_nx = torch.abs(cell_to_face_x(helpers["boundary_nx"]))
        abs_boundary_ny = torch.abs(cell_to_face_y(helpers["boundary_ny"]))

        base_flux_x = coeff_x * grad_u_x
        base_flux_y = coeff_y * grad_u_y

        tau_bulk_x = beta_bulk_x * (h**2) * coeff_x * lap_grad_x
        tau_bulk_y = beta_bulk_y * (h**2) * coeff_y * lap_grad_y

        if disable_interface_correction:
            tau_int_x = torch.zeros_like(base_flux_x)
            tau_int_y = torch.zeros_like(base_flux_y)
        else:
            tau_int_x = rho_int_x * (beta_n_x * abs_normal_x + beta_t_x * (1.0 - abs_normal_x)) * base_flux_x
            tau_int_y = rho_int_y * (beta_n_y * abs_normal_y + beta_t_y * (1.0 - abs_normal_y)) * base_flux_y

        if disable_boundary_correction:
            tau_bdry_x = torch.zeros_like(base_flux_x)
            tau_bdry_y = torch.zeros_like(base_flux_y)
        else:
            tau_bdry_x = rho_bdry_x * beta_b_x * abs_boundary_nx * base_flux_x
            tau_bdry_y = rho_bdry_y * beta_b_y * abs_boundary_ny * base_flux_y

        tau_x = tau_bulk_x + tau_int_x + tau_bdry_x
        tau_y = tau_bulk_y + tau_int_y + tau_bdry_y

        return self._finalize_outputs(
            coeff,
            u_backbone,
            tau_x,
            tau_y,
            helpers,
            tau_bulk_x=tau_bulk_x,
            tau_bulk_y=tau_bulk_y,
            tau_int_x=tau_int_x,
            tau_int_y=tau_int_y,
            tau_bdry_x=tau_bdry_x,
            tau_bdry_y=tau_bdry_y,
            rho_int_x=rho_int_x,
            rho_int_y=rho_int_y,
            rho_bdry_x=rho_bdry_x,
            rho_bdry_y=rho_bdry_y,
        )

    def from_direct_flux(self, coeff: torch.Tensor, u_backbone: torch.Tensor, tau_cell: torch.Tensor) -> dict[str, torch.Tensor]:
        helpers = self._grid_helpers(coeff)
        tau_x = cell_to_face_x(tau_cell[..., 0])
        tau_y = cell_to_face_y(tau_cell[..., 1])
        zeros_x = torch.zeros_like(tau_x)
        zeros_y = torch.zeros_like(tau_y)
        return self._finalize_outputs(
            coeff,
            u_backbone,
            tau_x,
            tau_y,
            helpers,
            tau_bulk_x=tau_x,
            tau_bulk_y=tau_y,
            tau_int_x=zeros_x,
            tau_int_y=zeros_y,
            tau_bdry_x=zeros_x,
            tau_bdry_y=zeros_y,
            rho_int_x=torch.zeros_like(tau_x),
            rho_int_y=torch.zeros_like(tau_y),
            rho_bdry_x=torch.zeros_like(tau_x),
            rho_bdry_y=torch.zeros_like(tau_y),
        )

    def from_direct_bias(self, coeff: torch.Tensor, u_backbone: torch.Tensor, bias: torch.Tensor) -> dict[str, torch.Tensor]:
        helpers = self._grid_helpers(coeff)
        correction = enforce_zero_boundary(bias)
        u_corrected = enforce_zero_boundary(u_backbone + correction)
        h = 1.0 / float(coeff.shape[-1] - 1)

        q_backbone_x, q_backbone_y = darcy_flux(coeff, u_backbone, h)
        q_corrected_x, q_corrected_y = darcy_flux(coeff, u_corrected, h)
        residual_backbone = 1.0 - darcy_operator(coeff, u_backbone, h)
        residual_corrected = darcy_operator(coeff, u_corrected, h) - 1.0
        tau_x = torch.zeros((bias.shape[0], bias.shape[1] - 1, bias.shape[2]), device=bias.device, dtype=bias.dtype)
        tau_y = torch.zeros((bias.shape[0], bias.shape[1], bias.shape[2] - 1), device=bias.device, dtype=bias.dtype)

        return {
            "u_corrected": u_corrected,
            "b_h": correction,
            "tau_x": tau_x,
            "tau_y": tau_y,
            "tau_bulk_x": torch.zeros_like(tau_x),
            "tau_bulk_y": torch.zeros_like(tau_y),
            "tau_int_x": torch.zeros_like(tau_x),
            "tau_int_y": torch.zeros_like(tau_y),
            "tau_bdry_x": torch.zeros_like(tau_x),
            "tau_bdry_y": torch.zeros_like(tau_y),
            "rho_int_x": torch.zeros_like(tau_x),
            "rho_int_y": torch.zeros_like(tau_y),
            "rho_bdry_x": torch.zeros_like(tau_x),
            "rho_bdry_y": torch.zeros_like(tau_y),
            "residual_backbone": residual_backbone,
            "residual_corrected": residual_corrected,
            "flux_backbone_x": q_backbone_x,
            "flux_backbone_y": q_backbone_y,
            "flux_corrected_x": q_corrected_x,
            "flux_corrected_y": q_corrected_y,
            "interior_mask": helpers["interior_mask"],
            "x_face_mask": helpers["x_face_mask"],
            "y_face_mask": helpers["y_face_mask"],
            "d_boundary": helpers["d_boundary"],
        }


class DarcyFNOWithCorrection(nn.Module):
    def __init__(
        self,
        *,
        coeff_resolution: int,
        backbone_modes: int,
        backbone_width: int,
        correction_modes: int,
        correction_width: int,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        cg_max_iter: int,
        cg_tol: float,
        variant: str = "deterministic",
        ablation: str = "none",
        disable_interface_correction: bool = False,
        disable_boundary_correction: bool = False,
        normalizer_eps: float = 1e-5,
    ):
        super().__init__()
        self.coeff_resolution = int(coeff_resolution)
        self.backbone_modes = int(backbone_modes)
        self.backbone_width = int(backbone_width)
        self.correction_modes = int(correction_modes)
        self.correction_width = int(correction_width)
        self.variant = str(variant)
        self.ablation = str(ablation)
        self.disable_interface_correction = bool(disable_interface_correction)
        self.disable_boundary_correction = bool(disable_boundary_correction)
        self.normalizer_eps = float(normalizer_eps)

        self.backbone = BackboneWrapper(backbone_modes, backbone_width)
        self.correction_layer = StructuredFluxCorrectionLayer(cg_max_iter=cg_max_iter, cg_tol=cg_tol)
        self.correction_net = StructuredCorrectionNet(correction_modes, correction_width)

        self.latent_channels = latent_channels_for_ablation(self.ablation)
        self.direct_head = None
        self.bayesian_latent = None
        self.uncertainty_head = None
        if self.variant == "deterministic":
            if self.ablation != "none":
                self.direct_head = DeterministicLatentHead(correction_modes, correction_width, self.latent_channels)
        elif self.variant == "bayesian":
            self.bayesian_latent = BayesianLatentHead(correction_modes, correction_width, self.latent_channels)
            self.uncertainty_head = UncertaintyHead(correction_modes, correction_width)
        else:
            raise ValueError(f"Unsupported variant: {self.variant}")

        self.register_buffer("x_mean", x_mean.clone())
        self.register_buffer("x_std", x_std.clone())
        self.register_buffer("y_mean", y_mean.clone())
        self.register_buffer("y_std", y_std.clone())

    def encode_coeff(self, coeff: torch.Tensor) -> torch.Tensor:
        return (coeff - self.x_mean) / (self.x_std + self.normalizer_eps)

    def decode_solution(self, field: torch.Tensor) -> torch.Tensor:
        return field * (self.y_std + self.normalizer_eps) + self.y_mean

    def _backbone_input(self, coeff: torch.Tensor) -> torch.Tensor:
        coeff_norm = self.encode_coeff(coeff)
        grid = make_grid(coeff.shape[-1], coeff.device, coeff.dtype).unsqueeze(0).expand(coeff.shape[0], -1, -1, -1)
        return torch.cat([coeff_norm.unsqueeze(-1), grid], dim=-1)

    def _build_features(self, coeff: torch.Tensor, u_backbone: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        size = coeff.shape[-1]
        h = 1.0 / float(size - 1)
        residual_backbone = 1.0 - darcy_operator(coeff, u_backbone, h)
        lap = laplacian(u_backbone, h)
        grad_u_x, grad_u_y = centered_gradient(u_backbone, h)
        grad_log_a_x, grad_log_a_y = centered_gradient(torch.log(torch.clamp(coeff, min=1e-6)), h)
        grad_u_mag = torch.sqrt(grad_u_x.square() + grad_u_y.square() + 1e-12)
        grad_log_a_mag = torch.sqrt(grad_log_a_x.square() + grad_log_a_y.square() + 1e-12)
        d_boundary = boundary_distance(size, coeff.device, coeff.dtype).unsqueeze(0).expand(coeff.shape[0], -1, -1)

        features = torch.stack(
            [
                coeff,
                u_backbone,
                residual_backbone,
                lap,
                grad_u_mag,
                grad_log_a_mag,
                d_boundary,
            ],
            dim=-1,
        )
        aux = {
            "residual_backbone": residual_backbone,
            "lap": lap,
            "grad_u_mag": grad_u_mag,
            "grad_log_a_mag": grad_log_a_mag,
            "d_boundary": d_boundary,
        }
        return features, aux

    def _apply_latent(self, coeff: torch.Tensor, u_backbone: torch.Tensor, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.ablation == "none":
            return self.correction_layer.from_beta(
                coeff,
                u_backbone,
                latent,
                disable_interface_correction=self.disable_interface_correction,
                disable_boundary_correction=self.disable_boundary_correction,
            )
        if self.ablation == "direct-flux":
            return self.correction_layer.from_direct_flux(coeff, u_backbone, latent)
        if self.ablation == "direct-bias":
            return self.correction_layer.from_direct_bias(coeff, u_backbone, latent.squeeze(-1) if latent.shape[-1] == 1 else latent[..., 0])
        raise ValueError(f"Unsupported ablation: {self.ablation}")

    def forward(self, coeff: torch.Tensor, *, mc_samples: int = 1) -> dict[str, torch.Tensor]:
        backbone_pred_norm = self.backbone(self._backbone_input(coeff))
        u_backbone = enforce_zero_boundary(self.decode_solution(backbone_pred_norm))
        features, aux = self._build_features(coeff, u_backbone)

        if self.variant == "deterministic":
            if self.ablation == "none":
                latent = self.correction_net(features)
            else:
                latent = self.direct_head(features)
            correction_outputs = self._apply_latent(coeff, u_backbone, latent)
            correction_outputs.update(
                {
                    "u_backbone": u_backbone,
                    "beta": latent if self.ablation == "none" else None,
                    "beta_mu": latent if self.ablation == "none" else latent,
                    "beta_logvar": None,
                    "pred_mean": correction_outputs["u_corrected"],
                    "pred_std": torch.zeros_like(correction_outputs["u_corrected"]),
                    "pred_logvar": torch.zeros_like(correction_outputs["u_corrected"]),
                    "u_corrected_mean": correction_outputs["u_corrected"],
                    "u_corrected_samples": correction_outputs["u_corrected"].unsqueeze(0),
                    "sample_b_h": correction_outputs["b_h"].unsqueeze(0),
                    "sample_tau_x": correction_outputs["tau_x"].unsqueeze(0),
                    "sample_tau_y": correction_outputs["tau_y"].unsqueeze(0),
                }
            )
            correction_outputs.update(aux)
            return correction_outputs

        beta_mu, beta_logvar = self.bayesian_latent(features)
        pred_logvar = self.uncertainty_head(features)
        sample_count = max(int(mc_samples), 1)

        sample_outputs = []
        for _ in range(sample_count):
            epsilon = torch.randn_like(beta_mu)
            latent = beta_mu + torch.exp(0.5 * beta_logvar) * epsilon
            sample_outputs.append(self._apply_latent(coeff, u_backbone, latent))

        u_samples = torch.stack([item["u_corrected"] for item in sample_outputs], dim=0)
        b_samples = torch.stack([item["b_h"] for item in sample_outputs], dim=0)
        tau_x_samples = torch.stack([item["tau_x"] for item in sample_outputs], dim=0)
        tau_y_samples = torch.stack([item["tau_y"] for item in sample_outputs], dim=0)

        correction_outputs = {
            "u_backbone": u_backbone,
            "u_corrected": u_samples.mean(dim=0),
            "u_corrected_mean": u_samples.mean(dim=0),
            "u_corrected_samples": u_samples,
            "pred_mean": u_samples.mean(dim=0),
            "pred_logvar": pred_logvar,
            "pred_std": torch.exp(0.5 * pred_logvar),
            "beta_mu": beta_mu,
            "beta_logvar": beta_logvar,
            "beta": beta_mu,
            "b_h": b_samples.mean(dim=0),
            "sample_b_h": b_samples,
            "tau_x": tau_x_samples.mean(dim=0),
            "tau_y": tau_y_samples.mean(dim=0),
            "sample_tau_x": tau_x_samples,
            "sample_tau_y": tau_y_samples,
            "tau_bulk_x": torch.stack([item["tau_bulk_x"] for item in sample_outputs], dim=0).mean(dim=0),
            "tau_bulk_y": torch.stack([item["tau_bulk_y"] for item in sample_outputs], dim=0).mean(dim=0),
            "tau_int_x": torch.stack([item["tau_int_x"] for item in sample_outputs], dim=0).mean(dim=0),
            "tau_int_y": torch.stack([item["tau_int_y"] for item in sample_outputs], dim=0).mean(dim=0),
            "tau_bdry_x": torch.stack([item["tau_bdry_x"] for item in sample_outputs], dim=0).mean(dim=0),
            "tau_bdry_y": torch.stack([item["tau_bdry_y"] for item in sample_outputs], dim=0).mean(dim=0),
            "rho_int_x": sample_outputs[0]["rho_int_x"],
            "rho_int_y": sample_outputs[0]["rho_int_y"],
            "rho_bdry_x": sample_outputs[0]["rho_bdry_x"],
            "rho_bdry_y": sample_outputs[0]["rho_bdry_y"],
            "residual_backbone": sample_outputs[0]["residual_backbone"],
            "residual_corrected": torch.stack([item["residual_corrected"] for item in sample_outputs], dim=0).mean(dim=0),
            "flux_backbone_x": sample_outputs[0]["flux_backbone_x"],
            "flux_backbone_y": sample_outputs[0]["flux_backbone_y"],
            "flux_corrected_x": torch.stack([item["flux_corrected_x"] for item in sample_outputs], dim=0).mean(dim=0),
            "flux_corrected_y": torch.stack([item["flux_corrected_y"] for item in sample_outputs], dim=0).mean(dim=0),
            "interior_mask": sample_outputs[0]["interior_mask"],
            "x_face_mask": sample_outputs[0]["x_face_mask"],
            "y_face_mask": sample_outputs[0]["y_face_mask"],
            "d_boundary": sample_outputs[0]["d_boundary"],
        }
        correction_outputs.update(aux)
        return correction_outputs


def build_correction_model(
    *,
    coeff_resolution: int,
    backbone_modes: int,
    backbone_width: int,
    correction_modes: int,
    correction_width: int,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    cg_max_iter: int,
    cg_tol: float,
    variant: str = "deterministic",
    ablation: str = "none",
    disable_interface_correction: bool = False,
    disable_boundary_correction: bool = False,
) -> DarcyFNOWithCorrection:
    return DarcyFNOWithCorrection(
        coeff_resolution=coeff_resolution,
        backbone_modes=backbone_modes,
        backbone_width=backbone_width,
        correction_modes=correction_modes,
        correction_width=correction_width,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        cg_max_iter=cg_max_iter,
        cg_tol=cg_tol,
        variant=variant,
        ablation=ablation,
        disable_interface_correction=disable_interface_correction,
        disable_boundary_correction=disable_boundary_correction,
    )


def compute_losses(
    outputs: dict[str, torch.Tensor],
    coeff: torch.Tensor,
    target_coarse: torch.Tensor,
    target_fine: torch.Tensor,
    *,
    variant: str,
    disable_flux_loss: bool,
    lambda_backbone: float,
    lambda_state: float,
    lambda_pde: float,
    lambda_flux: float,
    lambda_reg: float,
    lambda_mask: float,
    lambda_nll: float = 1.0,
    lambda_kl_beta: float = 0.0,
    lambda_kl_var: float = 0.0,
) -> tuple[LossBundle, dict[str, torch.Tensor]]:
    size = coeff.shape[-1]
    h = 1.0 / float(size - 1)
    interior = outputs["interior_mask"].expand(coeff.shape[0], -1, -1)
    face_mask_x = outputs["x_face_mask"].expand(coeff.shape[0], -1, -1)
    face_mask_y = outputs["y_face_mask"].expand(coeff.shape[0], -1, -1)

    prediction = outputs["pred_mean"] if variant == "bayesian" else outputs["u_corrected"]
    q_fine_x, q_fine_y = darcy_flux(coeff, target_fine, h)
    flux_error_x = outputs["flux_corrected_x"] - q_fine_x
    flux_error_y = outputs["flux_corrected_y"] - q_fine_y

    loss_backbone = masked_mean_square(outputs["u_backbone"] - target_coarse, interior).mean()
    loss_state = masked_mean_square(prediction - target_fine, interior).mean()
    loss_pde = masked_mean_square(darcy_operator(coeff, prediction, h) - 1.0, interior).mean()
    loss_flux = masked_face_mean_square(flux_error_x, flux_error_y, face_mask_x, face_mask_y).mean()
    loss_reg = smooth_face_norm(outputs["tau_x"], outputs["tau_y"]).mean()
    loss_mask = (
        masked_face_mean_square(
            (1.0 - outputs["rho_int_x"]) * outputs["tau_int_x"],
            (1.0 - outputs["rho_int_y"]) * outputs["tau_int_y"],
            face_mask_x,
            face_mask_y,
        )
        + masked_face_mean_square(
            (1.0 - outputs["rho_bdry_x"]) * outputs["tau_bdry_x"],
            (1.0 - outputs["rho_bdry_y"]) * outputs["tau_bdry_y"],
            face_mask_x,
            face_mask_y,
        )
    ).mean()

    loss_nll = torch.zeros((), device=coeff.device, dtype=coeff.dtype)
    loss_kl_beta = torch.zeros((), device=coeff.device, dtype=coeff.dtype)
    loss_kl_var = torch.zeros((), device=coeff.device, dtype=coeff.dtype)

    total = (
        lambda_backbone * loss_backbone
        + lambda_state * loss_state
        + lambda_pde * loss_pde
        + (0.0 if disable_flux_loss else lambda_flux * loss_flux)
        + lambda_reg * loss_reg
        + lambda_mask * loss_mask
    )

    if variant == "bayesian":
        pred_logvar = outputs["pred_logvar"]
        pred_var = torch.exp(pred_logvar)
        nll_map = 0.5 * (((target_fine - prediction).square() / torch.clamp(pred_var, min=1e-6)) + pred_logvar)
        loss_nll = masked_mean(nll_map, interior).mean()
        loss_kl_beta = kl_standard_normal(outputs["beta_mu"], outputs["beta_logvar"]).mean()
        loss_kl_var = masked_mean(0.5 * (pred_var - 1.0 - pred_logvar), interior).mean()
        total = (
            lambda_backbone * loss_backbone
            + lambda_nll * loss_nll
            + lambda_pde * loss_pde
            + (0.0 if disable_flux_loss else lambda_flux * loss_flux)
            + lambda_reg * loss_reg
            + lambda_mask * loss_mask
            + lambda_kl_beta * loss_kl_beta
            + lambda_kl_var * loss_kl_var
        )

    diagnostics = {
        "fine_l2_corrected": relative_l2(prediction, target_fine),
        "fine_l2_backbone": relative_l2(outputs["u_backbone"], target_fine),
        "coarse_l2_backbone": relative_l2(outputs["u_backbone"], target_coarse),
        "flux_error_x": flux_error_x,
        "flux_error_y": flux_error_y,
        "flux_error_magnitude": face_to_cell_magnitude(flux_error_x, flux_error_y),
        "pred_std": outputs["pred_std"],
    }
    return (
        LossBundle(
            total=total,
            backbone_coarse=loss_backbone,
            state_fine=loss_state,
            pde=loss_pde,
            flux=loss_flux,
            reg=loss_reg,
            mask=loss_mask,
            nll=loss_nll,
            kl_beta=loss_kl_beta,
            kl_var=loss_kl_var,
        ),
        diagnostics,
    )
