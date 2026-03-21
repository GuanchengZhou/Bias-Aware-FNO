#!/usr/bin/env python3
"""Train an FNO on Darcy FEM data with coarse-grid supervision and fine-grid testing."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from timeit import default_timer

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from Adam import Adam
from run_artifacts import (
    append_csv_row,
    build_run_config,
    ensure_run_dir,
    log_message,
    standard_artifact_paths,
    write_json,
)
from utilities3 import LpLoss, MatReader, UnitGaussianNormalizer, count_params


REPO_ROOT = Path(__file__).resolve().parent
MPLCONFIGDIR = REPO_ROOT / "tmp" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt


def default_dataset_path(split: str, n_samples: int, coeff_resolution: int, coarse_n: int, fine_n: int) -> Path:
    return REPO_ROOT / "data" / f"darcy_fem_r{coeff_resolution}_C{coarse_n}_F{fine_n}_N{n_samples}_{split}.mat"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO2d(nn.Module):
    def __init__(self, modes1: int, modes2: int, width: int):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9
        self.fc0 = nn.Linear(3, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


def make_grid(size: int) -> torch.Tensor:
    coords = np.linspace(0.0, 1.0, size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(coords, coords, indexing="ij")
    return torch.tensor(np.stack([grid_x, grid_y], axis=-1), dtype=torch.float32)


def load_fields(path: Path, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reader = MatReader(str(path))
    coeff = reader.read_field("coeff")[:count]
    sol_coarse = reader.read_field("sol_coarse")[:count]
    sol_fine = reader.read_field("sol_fine")[:count]
    return coeff, sol_coarse, sol_fine


def build_inputs(coeff: torch.Tensor) -> torch.Tensor:
    grid = make_grid(coeff.shape[-1]).unsqueeze(0).repeat(coeff.shape[0], 1, 1, 1)
    return torch.cat([coeff.unsqueeze(-1), grid], dim=-1)


def format_experiment_name(train_path: Path, ntrain: int, epochs: int, modes: int, width: int) -> str:
    return f"{train_path.stem}_fourier_2d_darcy_N{ntrain}_ep{epochs}_m{modes}_w{width}"


def save_sample_plot(path: Path, coeff: torch.Tensor, pred: torch.Tensor, coarse: torch.Tensor, fine: torch.Tensor) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    panels = [
        (coeff, "Coefficient", "viridis"),
        (pred, "Prediction", "RdBu_r"),
        (coarse, "Coarse Target", "RdBu_r"),
        (fine, "Fine Target", "RdBu_r"),
        (fine - pred, "Fine - Pred", "RdBu_r"),
    ]
    for ax, (field, title, cmap) in zip(axes, panels):
        im = ax.imshow(field.cpu().numpy(), cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FNO-Darcy on coarse FEM targets and test on fine FEM targets.")
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--test-path", type=Path, default=None)
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=100)
    parser.add_argument("--coeff-resolution", type=int, default=256)
    parser.add_argument("--coarse-n", type=int, default=150)
    parser.add_argument("--fine-n", type=int, default=300)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scheduler-step", type=int, default=200)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-sample-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    train_path = args.train_path or default_dataset_path(
        "train",
        args.ntrain,
        args.coeff_resolution,
        args.coarse_n,
        args.fine_n,
    )
    test_path = args.test_path or default_dataset_path(
        "test",
        args.ntest,
        args.coeff_resolution,
        args.coarse_n,
        args.fine_n,
    )
    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")

    coeff_train, y_train_coarse, _ = load_fields(train_path, args.ntrain)
    coeff_test, y_test_coarse, y_test_fine = load_fields(test_path, args.ntest)
    coeff_test_raw = coeff_test.clone()

    x_normalizer = UnitGaussianNormalizer(coeff_train)
    coeff_train_norm = x_normalizer.encode(coeff_train)
    coeff_test_norm = x_normalizer.encode(coeff_test)

    y_normalizer = UnitGaussianNormalizer(y_train_coarse)

    x_train = build_inputs(coeff_train_norm)
    x_test = build_inputs(coeff_test_norm)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train_coarse),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, coeff_test_raw, y_test_coarse, y_test_fine),
        batch_size=args.batch_size,
        shuffle=False,
    )

    experiment_name = format_experiment_name(
        train_path=train_path,
        ntrain=args.ntrain,
        epochs=args.epochs,
        modes=args.modes,
        width=args.width,
    )
    run_dir = ensure_run_dir(REPO_ROOT, experiment_name)
    artifacts = standard_artifact_paths(run_dir)
    config = build_run_config(
        task="darcy_fem",
        script_name=Path(__file__).name,
        experiment_name=run_dir.name,
        run_dir=run_dir,
        train_path=train_path,
        test_path=test_path,
        args=args,
    )
    write_json(artifacts["config"], config)

    device = resolve_device(args.device)
    model = FNO2d(args.modes, args.modes, args.width).to(device)
    log_message(f"run_dir {run_dir}", artifacts["train_log"], tqdm_module=tqdm)
    log_message(
        f"train_x {tuple(x_train.shape)} train_y {tuple(y_train_coarse.shape)}",
        artifacts["train_log"],
        tqdm_module=tqdm,
    )
    log_message(
        f"test_x {tuple(x_test.shape)} test_y {tuple(y_test_fine.shape)}",
        artifacts["train_log"],
        tqdm_module=tqdm,
    )
    log_message(f"params {count_params(model)}", artifacts["train_log"], tqdm_module=tqdm)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step,
        gamma=args.scheduler_gamma,
    )
    loss_fn = LpLoss(size_average=False)
    if device.type == "cuda":
        y_normalizer.cuda()

    last_sample = None
    last_test_coarse_reference = float("nan")
    last_test_fine = float("nan")
    epoch_iterator = range(args.epochs)
    if tqdm is not None:
        epoch_iterator = tqdm(
            epoch_iterator,
            desc="Darcy FNO Epochs",
            dynamic_ncols=True,
            position=0,
        )

    for ep in epoch_iterator:
        model.train()
        t1 = default_timer()
        train_l2_coarse = 0.0
        train_count = 0

        train_iterator = train_loader
        if tqdm is not None:
            train_iterator = tqdm(
                train_loader,
                desc=f"Train {ep + 1:04d}/{args.epochs:04d}",
                total=len(train_loader),
                dynamic_ncols=True,
                position=1,
                leave=False,
            )

        for xx, yy_coarse in train_iterator:
            xx = xx.to(device)
            yy_coarse = yy_coarse.to(device)

            optimizer.zero_grad()
            pred = y_normalizer.decode(model(xx))
            loss = loss_fn(pred.reshape(pred.shape[0], -1), yy_coarse.reshape(yy_coarse.shape[0], -1))
            loss.backward()
            optimizer.step()

            train_l2_coarse += loss.item()
            train_count += yy_coarse.shape[0]

            if tqdm is not None:
                train_iterator.set_postfix(
                    batch_l2=f"{loss.item() / yy_coarse.shape[0]:.3e}",
                    avg_l2=f"{train_l2_coarse / train_count:.3e}",
                )

        scheduler.step()
        t2 = default_timer()
        train_metric = train_l2_coarse / train_count
        should_test = ((ep + 1) % args.test_every == 0) or (ep == args.epochs - 1)

        train_summary = f"{ep:04d} {t2 - t1:.4f} {train_metric:.6e}"
        append_csv_row(
            artifacts["train_metrics"],
            ["epoch", "time_sec", "train_l2_coarse"],
            {
                "epoch": ep,
                "time_sec": t2 - t1,
                "train_l2_coarse": train_metric,
            },
        )
        log_message(train_summary, artifacts["train_log"], tqdm_module=tqdm)

        if should_test:
            model.eval()
            test_l2_coarse_reference = 0.0
            test_l2_fine = 0.0
            test_count = 0
            test_iterator = test_loader
            if tqdm is not None:
                test_iterator = tqdm(
                    test_loader,
                    desc=f"Test  {ep + 1:04d}/{args.epochs:04d}",
                    total=len(test_loader),
                    dynamic_ncols=True,
                    position=1,
                    leave=False,
                )
            with torch.no_grad():
                for xx, coeff_batch, yy_coarse, yy_fine in test_iterator:
                    xx = xx.to(device)
                    yy_coarse = yy_coarse.to(device)
                    yy_fine = yy_fine.to(device)
                    pred = y_normalizer.decode(model(xx))

                    test_l2_coarse_reference += loss_fn(
                        pred.reshape(pred.shape[0], -1),
                        yy_coarse.reshape(yy_coarse.shape[0], -1),
                    ).item()
                    test_l2_fine += loss_fn(
                        pred.reshape(pred.shape[0], -1),
                        yy_fine.reshape(yy_fine.shape[0], -1),
                    ).item()
                    test_count += yy_coarse.shape[0]
                    last_sample = (
                        coeff_batch[0],
                        pred[0].cpu(),
                        yy_coarse[0].cpu(),
                        yy_fine[0].cpu(),
                    )

                    if tqdm is not None:
                        test_iterator.set_postfix(
                            coarse_l2=f"{test_l2_coarse_reference / test_count:.3e}",
                            fine_l2=f"{test_l2_fine / test_count:.3e}",
                        )

            last_test_coarse_reference = test_l2_coarse_reference / test_count
            last_test_fine = test_l2_fine / test_count
            test_summary = (
                f"{ep:04d} {t2 - t1:.4f} "
                f"{train_metric:.6e} "
                f"{last_test_coarse_reference:.6e} "
                f"{last_test_fine:.6e}"
            )
            if tqdm is not None:
                tqdm.write(test_summary)
            else:
                print(test_summary)
            with artifacts["train_log"].open("a", encoding="utf-8") as handle:
                handle.write(test_summary + "\n")
            append_csv_row(
                artifacts["test_metrics"],
                ["epoch", "time_sec", "train_l2_coarse", "test_l2_coarse_reference", "test_l2_fine"],
                {
                    "epoch": ep,
                    "time_sec": t2 - t1,
                    "train_l2_coarse": train_metric,
                    "test_l2_coarse_reference": last_test_coarse_reference,
                    "test_l2_fine": last_test_fine,
                },
            )

        if tqdm is not None:
            epoch_iterator.set_postfix(
                train_l2_coarse=f"{train_metric:.3e}",
                test_l2_fine=f"{last_test_fine:.3e}" if np.isfinite(last_test_fine) else "pending",
            )
    torch.save(model, artifacts["model"])
    torch.save(model.state_dict(), artifacts["model_state_dict"])
    log_message(f"Saved model -> {artifacts['model']}", artifacts["train_log"], tqdm_module=tqdm)
    log_message(
        f"Saved model_state_dict -> {artifacts['model_state_dict']}",
        artifacts["train_log"],
        tqdm_module=tqdm,
    )

    if args.save_sample_plot and last_sample is not None:
        coeff_plot, pred_plot, coarse_plot, fine_plot = last_sample
        save_sample_plot(artifacts["sample"], coeff_plot, pred_plot, coarse_plot, fine_plot)
        log_message(f"Saved sample plot -> {artifacts['sample']}", artifacts["train_log"], tqdm_module=tqdm)

    scipy.io.savemat(
        artifacts["train_summary"],
        {
            "train_x_shape": np.asarray(x_train.shape, dtype=np.int64),
            "train_y_shape": np.asarray(y_train_coarse.shape, dtype=np.int64),
            "test_x_shape": np.asarray(x_test.shape, dtype=np.int64),
            "test_y_shape": np.asarray(y_test_fine.shape, dtype=np.int64),
            "coeff_resolution": np.asarray([args.coeff_resolution], dtype=np.int64),
        },
    )
    log_message(f"Saved train summary -> {artifacts['train_summary']}", artifacts["train_log"], tqdm_module=tqdm)

    eval_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_2d_darcy_fem.py"),
        "--run-dir",
        str(run_dir),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    log_message(
        f"Running evaluation -> {' '.join(eval_command)}",
        artifacts["train_log"],
        tqdm_module=tqdm,
    )
    subprocess.run(eval_command, check=True, cwd=REPO_ROOT)
    log_message("Evaluation completed.", artifacts["train_log"], tqdm_module=tqdm)


if __name__ == "__main__":
    main()
