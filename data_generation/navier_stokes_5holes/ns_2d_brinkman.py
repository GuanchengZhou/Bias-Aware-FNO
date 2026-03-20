#!/usr/bin/env python3
"""Generate random five-hole Brinkman-penalized Navier-Stokes data in FNO style."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.navier_stokes_5holes.random_fields import GaussianRF


def format_float_token(value: float) -> str:
    return f"{value:.0e}".replace("+0", "").replace("+", "")


def default_file_name(split: str, n_samples: int, nu: float, eta: float, record_steps: int, resolution: int) -> str:
    return (
        f"ns_5holes_V{format_float_token(nu)}_E{format_float_token(eta)}_"
        f"N{n_samples}_T{record_steps}_R{resolution}_{split}.mat"
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Brinkman-penalized Navier-Stokes datasets with five random holes.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--n-ood", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--record-steps", type=int, default=20)
    parser.add_argument("--final-time", type=float, default=4.0)
    parser.add_argument("--delta-t", type=float, default=5e-4)
    parser.add_argument("--nu", type=float, default=1e-3)
    parser.add_argument("--eta", type=float, default=5e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--n-holes", type=int, default=5)
    parser.add_argument("--radius-min", type=float, default=0.05)
    parser.add_argument("--radius-max", type=float, default=0.08)
    parser.add_argument("--boundary-margin", type=float, default=0.05)
    parser.add_argument("--hole-gap-min", type=float, default=0.03)
    parser.add_argument("--smooth-sigma", type=float, default=1.5, help="Gaussian smoothing sigma in pixels.")
    parser.add_argument("--max-placement-attempts", type=int, default=2000)
    parser.add_argument("--grf-alpha", type=float, default=2.5)
    parser.add_argument("--grf-tau", type=float, default=7.0)
    parser.add_argument("--ood-radius-min", type=float, default=None)
    parser.add_argument("--ood-radius-max", type=float, default=None)
    parser.add_argument("--ood-hole-gap-min", type=float, default=None)
    return parser.parse_args()


def make_grid(resolution: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.linspace(0.0, 1.0, resolution + 1, device=device, dtype=dtype)[:-1]
    return torch.meshgrid(coords, coords, indexing="ij")


def sample_theta(
    rng: np.random.Generator,
    n_holes: int,
    radius_range: tuple[float, float],
    boundary_margin: float,
    hole_gap_min: float,
    max_attempts: int,
) -> np.ndarray:
    centers: list[tuple[float, float, float]] = []
    radius_min, radius_max = radius_range
    for _ in range(n_holes):
        placed = False
        for _attempt in range(max_attempts):
            radius = float(rng.uniform(radius_min, radius_max))
            low = boundary_margin + radius
            high = 1.0 - boundary_margin - radius
            if low >= high:
                raise ValueError("Geometry constraints are infeasible; reduce boundary margin or radius range.")
            cx = float(rng.uniform(low, high))
            cy = float(rng.uniform(low, high))
            ok = True
            for ox, oy, oradius in centers:
                if math.hypot(cx - ox, cy - oy) < radius + oradius + hole_gap_min:
                    ok = False
                    break
            if ok:
                centers.append((cx, cy, radius))
                placed = True
                break
        if not placed:
            raise RuntimeError("Failed to sample non-overlapping holes; try smaller radii or smaller minimum gap.")
    return np.asarray(centers, dtype=np.float64)


def sample_theta_batch(
    n_samples: int,
    rng: np.random.Generator,
    n_holes: int,
    radius_range: tuple[float, float],
    boundary_margin: float,
    hole_gap_min: float,
    max_attempts: int,
) -> np.ndarray:
    theta = np.empty((n_samples, n_holes, 3), dtype=np.float64)
    for idx in range(n_samples):
        theta[idx] = sample_theta(rng, n_holes, radius_range, boundary_margin, hole_gap_min, max_attempts)
    return theta


def build_masks(
    theta: np.ndarray,
    resolution: int,
    device: torch.device,
    dtype: torch.dtype,
    smooth_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid_x, grid_y = make_grid(resolution, device=device, dtype=dtype)
    grid_x = grid_x.unsqueeze(0).unsqueeze(0)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0)

    theta_t = torch.as_tensor(theta, device=device, dtype=dtype)
    cx = theta_t[:, :, 0].unsqueeze(-1).unsqueeze(-1)
    cy = theta_t[:, :, 1].unsqueeze(-1).unsqueeze(-1)
    radius = theta_t[:, :, 2].unsqueeze(-1).unsqueeze(-1)

    dist2 = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
    mask = torch.any(dist2 <= radius**2, dim=1).to(dtype)

    if smooth_sigma <= 0:
        chi_smooth = mask.clone()
    else:
        chi_np = mask.detach().cpu().numpy()
        smooth_np = np.stack(
            [gaussian_filter(chi_np[i], sigma=float(smooth_sigma), mode="nearest") for i in range(chi_np.shape[0])],
            axis=0,
        )
        smooth_np = np.clip(smooth_np, 0.0, 1.0)
        chi_smooth = torch.as_tensor(smooth_np, device=device, dtype=dtype)
    return mask, chi_smooth


def build_forcing(resolution: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x, y = make_grid(resolution, device=device, dtype=dtype)
    return 0.1 * (torch.sin(2 * math.pi * (x + y)) + torch.cos(2 * math.pi * (x + y)))


class BrinkmanNavierStokes2d(object):
    def __init__(self, resolution: int, device: torch.device, dtype: torch.dtype):
        self.resolution = resolution
        self.device = device
        self.dtype = dtype

        freq_idx = torch.fft.fftfreq(resolution, d=1.0 / resolution, device=device, dtype=dtype)
        freq_phys = 2.0 * math.pi * freq_idx
        kx, ky = torch.meshgrid(freq_phys, freq_phys, indexing="ij")
        idx_x, idx_y = torch.meshgrid(freq_idx, freq_idx, indexing="ij")

        self.kx = kx.unsqueeze(0)
        self.ky = ky.unsqueeze(0)
        self.lap = (kx**2 + ky**2).unsqueeze(0)
        self.inv_lap = torch.zeros_like(self.lap)
        nonzero = self.lap != 0
        self.inv_lap[nonzero] = 1.0 / self.lap[nonzero]

        cutoff = resolution / 3.0
        dealias = (idx_x.abs() <= cutoff) & (idx_y.abs() <= cutoff)
        self.dealias = dealias.unsqueeze(0).to(dtype)

    def stream_function(self, w_hat: torch.Tensor) -> torch.Tensor:
        return w_hat * self.inv_lap

    def velocity_field(self, w_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        psi_hat = self.stream_function(w_hat)
        ux_hat = 1j * self.ky * psi_hat
        uy_hat = -1j * self.kx * psi_hat
        ux = torch.fft.ifft2(ux_hat, dim=(-2, -1)).real
        uy = torch.fft.ifft2(uy_hat, dim=(-2, -1)).real
        return ux, uy

    def vorticity_gradient(self, w_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        wx = torch.fft.ifft2(1j * self.kx * w_hat, dim=(-2, -1)).real
        wy = torch.fft.ifft2(1j * self.ky * w_hat, dim=(-2, -1)).real
        return wx, wy

    def curl_weighted_velocity(self, ux: torch.Tensor, uy: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
        chi_ux_hat = torch.fft.fft2(chi * ux, dim=(-2, -1))
        chi_uy_hat = torch.fft.fft2(chi * uy, dim=(-2, -1))
        return torch.fft.ifft2(1j * self.kx * chi_uy_hat - 1j * self.ky * chi_ux_hat, dim=(-2, -1)).real

    def rhs(self, w_hat: torch.Tensor, forcing_hat: torch.Tensor, chi: torch.Tensor, eta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ux, uy = self.velocity_field(w_hat)
        wx, wy = self.vorticity_gradient(w_hat)
        adv_hat = torch.fft.fft2(ux * wx + uy * wy, dim=(-2, -1))
        penalty = self.curl_weighted_velocity(ux, uy, chi)
        penalty_hat = torch.fft.fft2(penalty, dim=(-2, -1))
        rhs = -adv_hat + forcing_hat - penalty_hat / eta
        rhs = rhs * self.dealias
        return rhs, ux, uy

    def solve(
        self,
        w0: torch.Tensor,
        forcing: torch.Tensor,
        chi: torch.Tensor,
        nu: float,
        eta: float,
        final_time: float,
        delta_t: float,
        record_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps = int(math.ceil(final_time / delta_t))
        if steps < record_steps:
            raise ValueError("record_steps cannot exceed the number of internal solver steps.")

        record_targets = np.rint(np.linspace(1, steps, record_steps)).astype(np.int64)
        if len(np.unique(record_targets)) != record_steps:
            raise ValueError("record_steps is too large for the chosen final_time and delta_t.")

        w_hat = torch.fft.fft2(w0, dim=(-2, -1))
        forcing_hat = torch.fft.fft2(forcing, dim=(-2, -1)).unsqueeze(0)
        forcing_hat = forcing_hat.expand(w0.shape[0], -1, -1)

        sol = torch.zeros((*w0.shape, record_steps), device=self.device, dtype=self.dtype)
        sol_t = torch.zeros(record_steps, device=self.device, dtype=self.dtype)

        target_idx = 0
        time = 0.0
        for step_idx in range(steps):
            rhs1, _, _ = self.rhs(w_hat, forcing_hat, chi, eta)
            denom = 1.0 + 0.5 * delta_t * nu * self.lap
            w_hat_tilde = (w_hat + delta_t * (rhs1 - 0.5 * nu * self.lap * w_hat)) / denom

            rhs2, _, _ = self.rhs(w_hat_tilde, forcing_hat, chi, eta)
            w_hat = (w_hat + delta_t * (0.5 * (rhs1 + rhs2) - 0.5 * nu * self.lap * w_hat)) / denom
            w_hat = w_hat * self.dealias
            w_hat[:, 0, 0] = 0.0

            time += delta_t
            if step_idx + 1 == record_targets[target_idx]:
                sol[..., target_idx] = torch.fft.ifft2(w_hat, dim=(-2, -1)).real
                sol_t[target_idx] = time
                target_idx += 1
                if target_idx == record_steps:
                    break

        return sol, sol_t


def build_split_spec(args: argparse.Namespace) -> list[tuple[str, int, tuple[float, float], float]]:
    specs = [
        ("train", int(args.n_train), (float(args.radius_min), float(args.radius_max)), float(args.hole_gap_min)),
        ("test", int(args.n_test), (float(args.radius_min), float(args.radius_max)), float(args.hole_gap_min)),
    ]
    if int(args.n_ood) > 0:
        ood_radius_min = float(args.ood_radius_min if args.ood_radius_min is not None else args.radius_min * 1.1)
        ood_radius_max = float(args.ood_radius_max if args.ood_radius_max is not None else args.radius_max * 1.2)
        ood_gap = float(args.ood_hole_gap_min if args.ood_hole_gap_min is not None else max(0.0, args.hole_gap_min * 0.75))
        specs.append(("ood", int(args.n_ood), (ood_radius_min, ood_radius_max), ood_gap))
    return specs


def save_split(
    split: str,
    out_path: Path,
    a: np.ndarray,
    u: np.ndarray,
    mask: np.ndarray,
    theta: np.ndarray,
    t: np.ndarray,
    forcing: np.ndarray,
    chi_smooth: np.ndarray,
    args: argparse.Namespace,
) -> None:
    payload = {
        "a": a.astype(np.float32),
        "u": u.astype(np.float32),
        "mask": mask.astype(np.float32),
        "theta": theta.astype(np.float32),
        "t": t.astype(np.float32),
        "f": forcing.astype(np.float32),
        "chi_smooth": chi_smooth.astype(np.float32),
        "nu": np.asarray([args.nu], dtype=np.float32),
        "eta": np.asarray([args.eta], dtype=np.float32),
        "final_time": np.asarray([args.final_time], dtype=np.float32),
        "delta_t": np.asarray([args.delta_t], dtype=np.float32),
        "split": np.asarray([split], dtype=object),
    }
    scipy.io.savemat(out_path, payload)


def generate_split(
    split: str,
    n_samples: int,
    radius_range: tuple[float, float],
    hole_gap_min: float,
    args: argparse.Namespace,
    solver: BrinkmanNavierStokes2d,
    grf: GaussianRF,
    forcing: torch.Tensor,
    rng: np.random.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> Path:
    theta = sample_theta_batch(
        n_samples=n_samples,
        rng=rng,
        n_holes=int(args.n_holes),
        radius_range=radius_range,
        boundary_margin=float(args.boundary_margin),
        hole_gap_min=hole_gap_min,
        max_attempts=int(args.max_placement_attempts),
    )
    mask, chi_smooth = build_masks(theta, int(args.resolution), device=device, dtype=dtype, smooth_sigma=float(args.smooth_sigma))
    w0 = grf.sample(n_samples).to(device=device, dtype=dtype)

    sol = torch.zeros((n_samples, int(args.resolution), int(args.resolution), int(args.record_steps)), device=device, dtype=dtype)
    sol_t = None
    for start in range(0, n_samples, int(args.batch_size)):
        end = min(start + int(args.batch_size), n_samples)
        print(f"{start}-{end} / {n_samples}")
        batch_sol, batch_t = solver.solve(
            w0[start:end],
            forcing,
            chi_smooth[start:end],
            nu=float(args.nu),
            eta=float(args.eta),
            final_time=float(args.final_time),
            delta_t=float(args.delta_t),
            record_steps=int(args.record_steps),
        )
        sol[start:end] = batch_sol
        if sol_t is None:
            sol_t = batch_t
        print(f"[{split}] generated {end}/{n_samples}")

    out_path = args.output_dir / default_file_name(split, n_samples, float(args.nu), float(args.eta), int(args.record_steps), int(args.resolution))
    save_split(
        split=split,
        out_path=out_path,
        a=w0.detach().cpu().numpy(),
        u=sol.detach().cpu().numpy(),
        mask=mask.detach().cpu().numpy(),
        theta=theta,
        t=sol_t.detach().cpu().numpy(),
        forcing=forcing.detach().cpu().numpy(),
        chi_smooth=chi_smooth.detach().cpu().numpy(),
        args=args,
    )
    return out_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = resolve_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    print(f"Using device={device}, dtype={dtype}, output_dir={args.output_dir}")

    grf = GaussianRF(2, int(args.resolution), alpha=float(args.grf_alpha), tau=float(args.grf_tau), device=device, dtype=dtype)
    forcing = build_forcing(int(args.resolution), device=device, dtype=dtype)
    solver = BrinkmanNavierStokes2d(int(args.resolution), device=device, dtype=dtype)
    rng = np.random.default_rng(int(args.seed))

    for split, n_samples, radius_range, hole_gap_min in build_split_spec(args):
        if n_samples <= 0:
            continue
        out_path = generate_split(
            split=split,
            n_samples=n_samples,
            radius_range=radius_range,
            hole_gap_min=hole_gap_min,
            args=args,
            solver=solver,
            grf=grf,
            forcing=forcing,
            rng=rng,
            device=device,
            dtype=dtype,
        )
        print(f"Saved {split} dataset -> {out_path}")


if __name__ == "__main__":
    main()
