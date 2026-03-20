#!/usr/bin/env python3
"""FNO-style recurrent training for random five-hole Navier-Stokes data."""

from __future__ import annotations

import argparse
from pathlib import Path
from timeit import default_timer

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F

from Adam import Adam
from utilities3 import LpLoss, MatReader, count_params


REPO_ROOT = Path(__file__).resolve().parent


def format_float_token(value: float) -> str:
    return f"{value:.0e}".replace("+0", "").replace("+", "")


def default_dataset_path(split: str, n_samples: int, nu: float, eta: float, record_steps: int, resolution: int) -> Path:
    name = (
        f"ns_5holes_V{format_float_token(nu)}_E{format_float_token(eta)}_"
        f"N{n_samples}_T{record_steps}_R{resolution}_{split}.mat"
    )
    return REPO_ROOT / "data" / name


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
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

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
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
    def __init__(self, modes1, modes2, width, t_in):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.t_in = t_in
        self.fc0 = nn.Linear(t_in + 1 + 2, self.width)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float32, device=device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float32, device=device)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

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

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


def rollout_autoregressive(model, xx, yy, t_in, step, loss_fn):
    total_loss = 0.0
    pred = None
    current = xx

    for t in range(0, yy.shape[-1], step):
        target = yy[..., t : t + step]
        im = model(current)
        total_loss = total_loss + loss_fn(im.reshape(im.shape[0], -1), target.reshape(target.shape[0], -1))
        pred = im if pred is None else torch.cat((pred, im), dim=-1)

        history = current[..., :t_in]
        mask = current[..., t_in : t_in + 1]
        history = torch.cat((history[..., step:], im), dim=-1)
        current = torch.cat((history, mask), dim=-1)

    return pred, total_loss


def format_experiment_name(train_path: Path, ntrain: int, epochs: int, modes: int, width: int, t_in: int, t_out: int) -> str:
    stem = train_path.stem
    return f"{stem}_fourier_2d_rnn_N{ntrain}_Tin{t_in}_Tout{t_out}_ep{epochs}_m{modes}_w{width}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an FNO recurrent model on random five-hole Navier-Stokes data.")
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--test-path", type=Path, default=None)
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--nu", type=float, default=1e-3)
    parser.add_argument("--eta", type=float, default=5e-3)
    parser.add_argument("--record-steps", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--sub", type=int, default=4)
    parser.add_argument("--T-in", type=int, default=10)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scheduler-step", type=int, default=100)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-sample-plot", action="store_true")
    return parser.parse_args()


def load_dataset(path: Path, count: int, sub: int, t_in: int, t_out: int):
    reader = MatReader(str(path))
    u = reader.read_field("u")[:count, ::sub, ::sub, : t_in + t_out]
    mask = reader.read_field("mask")[:count, ::sub, ::sub]
    a = u[..., :t_in]
    y = u[..., t_in : t_in + t_out]
    x = torch.cat((a, mask.unsqueeze(-1)), dim=-1)
    return x, y, mask


def ensure_output_dirs():
    for name in ["model", "results", "pred", "image"]:
        (REPO_ROOT / name).mkdir(parents=True, exist_ok=True)


def save_sample_plot(path: Path, pred: torch.Tensor, truth: torch.Tensor) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im0 = axes[0].imshow(truth.cpu().numpy(), cmap="RdBu_r")
    axes[0].set_title("Target")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    im1 = axes[1].imshow(pred.cpu().numpy(), cmap="RdBu_r")
    axes[1].set_title("Prediction")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    train_path = args.train_path or default_dataset_path("train", args.ntrain, args.nu, args.eta, args.record_steps, args.resolution)
    test_path = args.test_path or default_dataset_path("test", args.ntest, args.nu, args.eta, args.record_steps, args.resolution)
    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")

    ensure_output_dirs()
    device = resolve_device(args.device)

    train_x, train_y, _ = load_dataset(train_path, args.ntrain, args.sub, args.T_in, args.T)
    test_x, test_y, _ = load_dataset(test_path, args.ntest, args.sub, args.T_in, args.T)

    size = train_x.shape[1]
    print("train_x", tuple(train_x.shape), "train_y", tuple(train_y.shape))
    print("test_x", tuple(test_x.shape), "test_y", tuple(test_y.shape))

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_x, test_y), batch_size=args.batch_size, shuffle=False
    )

    model = FNO2d(args.modes, args.modes, args.width, args.T_in).to(device)
    print("params", count_params(model))

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
    )
    loss_fn = LpLoss(size_average=False)

    experiment_name = format_experiment_name(train_path, args.ntrain, args.epochs, args.modes, args.width, args.T_in, args.T)
    path_model = REPO_ROOT / "model" / experiment_name
    path_train_err = REPO_ROOT / "results" / f"{experiment_name}_train.txt"
    path_test_err = REPO_ROOT / "results" / f"{experiment_name}_test.txt"
    path_image = REPO_ROOT / "image" / f"{experiment_name}_sample.png"

    last_test_pred = None
    last_test_truth = None

    for ep in range(args.epochs):
        model.train()
        t1 = default_timer()
        train_l2_step = 0.0
        train_l2_full = 0.0
        train_count = 0

        for xx, yy in train_loader:
            xx = xx.to(device)
            yy = yy.to(device)

            pred, loss = rollout_autoregressive(model, xx, yy, args.T_in, args.step, loss_fn)
            l2_full = loss_fn(pred.reshape(pred.shape[0], -1), yy.reshape(yy.shape[0], -1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_l2_step += loss.item()
            train_l2_full += l2_full.item()
            train_count += yy.shape[0]

        model.eval()
        test_l2_step = 0.0
        test_l2_full = 0.0
        test_count = 0
        with torch.no_grad():
            for xx, yy in test_loader:
                xx = xx.to(device)
                yy = yy.to(device)
                pred, loss = rollout_autoregressive(model, xx, yy, args.T_in, args.step, loss_fn)
                l2_full = loss_fn(pred.reshape(pred.shape[0], -1), yy.reshape(yy.shape[0], -1))
                test_l2_step += loss.item()
                test_l2_full += l2_full.item()
                test_count += yy.shape[0]
                last_test_pred = pred
                last_test_truth = yy

        scheduler.step()
        t2 = default_timer()
        summary = (
            f"{ep:04d} {t2 - t1:.4f} "
            f"{train_l2_step / train_count:.6e} {train_l2_full / train_count:.6e} "
            f"{test_l2_step / test_count:.6e} {test_l2_full / test_count:.6e}"
        )
        print(summary)
        with path_train_err.open("a", encoding="utf-8") as f:
            f.write(summary + "\n")
        with path_test_err.open("a", encoding="utf-8") as f:
            f.write(summary + "\n")

    torch.save(model, path_model)
    print(f"Saved model -> {path_model}")

    if args.save_sample_plot and last_test_pred is not None and last_test_truth is not None:
        save_sample_plot(path_image, last_test_pred[0, :, :, -1], last_test_truth[0, :, :, -1])
        print(f"Saved sample plot -> {path_image}")

    scipy.io.savemat(
        REPO_ROOT / "pred" / f"{experiment_name}_train_summary.mat",
        {
            "train_x_shape": np.asarray(train_x.shape, dtype=np.int64),
            "train_y_shape": np.asarray(train_y.shape, dtype=np.int64),
            "test_x_shape": np.asarray(test_x.shape, dtype=np.int64),
            "test_y_shape": np.asarray(test_y.shape, dtype=np.int64),
            "spatial_size": np.asarray([size], dtype=np.int64),
        },
    )


if __name__ == "__main__":
    main()
