#!/usr/bin/env python3
"""Evaluate a trained FNO model on random five-hole Navier-Stokes data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fourier_2d_time_5holes import (
    FNO2d,
    SpectralConv2d_fast,
    default_dataset_path,
    format_experiment_name,
    resolve_device,
    rollout_autoregressive,
)
from run_artifacts import load_json, log_message, resolve_model_artifact, standard_artifact_paths, write_json
from utilities3 import LpLoss, MatReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained FNO model on random five-hole Navier-Stokes data.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
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
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-name", type=str, default=None)
    return parser.parse_args()


def load_eval_tensors(path: Path, count: int, sub: int, t_in: int, t_out: int):
    reader = MatReader(str(path))
    u = reader.read_field("u")[:count, ::sub, ::sub, : t_in + t_out]
    mask = reader.read_field("mask")[:count, ::sub, ::sub]
    theta = reader.read_field("theta")[:count]
    times = reader.read_field("t")
    x = torch.cat((u[..., :t_in], mask.unsqueeze(-1)), dim=-1)
    y = u[..., t_in : t_in + t_out]
    return x, y, mask, theta, times


def masked_relative_l2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    fluid = (1.0 - mask).unsqueeze(-1)
    diff = ((pred - target) * fluid).reshape(pred.shape[0], -1)
    ref = (target * fluid).reshape(target.shape[0], -1)
    diff_norm = torch.norm(diff, dim=1)
    ref_norm = torch.norm(ref, dim=1).clamp_min(1e-12)
    return diff_norm / ref_norm


def hydrate_args_from_run_config(args: argparse.Namespace, config: dict) -> Path:
    stored_args = config.get("args", {})
    for key in [
        "ntrain",
        "ntest",
        "nu",
        "eta",
        "record_steps",
        "resolution",
        "sub",
        "T_in",
        "T",
        "step",
        "modes",
        "width",
        "epochs",
    ]:
        if key in stored_args:
            setattr(args, key, stored_args[key])
    return args.test_path or Path(config["test_path"])


def load_model(model_path: Path, model_kind: str, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if model_kind == "full":
        model = torch.load(model_path, map_location=device, weights_only=False)
        return model.to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model = FNO2d(args.modes, args.modes, args.width, args.T_in).to(device)
    model.load_state_dict(state_dict)
    return model


def infer_model_kind(model_path: Path) -> str:
    return "state_dict" if model_path.name.endswith("state_dict.pt") else "full"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    eval_log_path = None
    eval_metrics_path = None

    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        artifacts = standard_artifact_paths(run_dir)
        config_path = artifacts["config"]
        if not config_path.exists():
            raise FileNotFoundError(f"Run config not found: {config_path}")
        config = load_json(config_path)
        test_path = hydrate_args_from_run_config(args, config)
        eval_log_path = artifacts["eval_log"]
        eval_metrics_path = artifacts["eval_metrics"]
        output_path = artifacts["predictions"]
        if args.model_path is None:
            model_path, model_kind = resolve_model_artifact(run_dir)
        else:
            model_path = args.model_path
            model_kind = infer_model_kind(model_path)
    else:
        run_dir = None
        test_path = args.test_path or default_dataset_path("test", args.ntest, args.nu, args.eta, args.record_steps, args.resolution)
        if args.model_path is None:
            train_path = default_dataset_path("train", args.ntrain, args.nu, args.eta, args.record_steps, args.resolution)
            exp_name = format_experiment_name(train_path, args.ntrain, args.epochs, args.modes, args.width, args.T_in, args.T)
            model_path = REPO_ROOT / "model" / exp_name
        else:
            model_path = args.model_path
        model_kind = infer_model_kind(model_path)
        output_name = args.output_name or f"{model_path.name}_eval"
        output_path = REPO_ROOT / "pred" / f"{output_name}.mat"
        output_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(message: str) -> None:
        if eval_log_path is not None:
            log_message(message, eval_log_path)
        else:
            print(message)

    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    emit(f"run_dir {run_dir}" if run_dir is not None else "run_dir <legacy>")
    emit(f"test_path {test_path}")
    emit(f"model_path {model_path}")

    x, y, mask, theta, times = load_eval_tensors(test_path, args.ntest, args.sub, args.T_in, args.T)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y, mask), batch_size=1, shuffle=False
    )

    model = load_model(model_path, model_kind, args, device)
    model.eval()

    loss_fn = LpLoss(size_average=False)
    pred = torch.zeros_like(y)
    full_scores = []
    fluid_scores = []

    with torch.no_grad():
        index = 0
        for xx, yy, mm in loader:
            xx = xx.to(device)
            yy = yy.to(device)
            mm = mm.to(device)
            out, _ = rollout_autoregressive(model, xx, yy, args.T_in, args.step, loss_fn)
            pred[index : index + xx.shape[0]] = out.cpu()
            full_value = loss_fn(out.reshape(out.shape[0], -1), yy.reshape(yy.shape[0], -1)).item()
            fluid_value = masked_relative_l2(out, yy, mm).mean().item()
            full_scores.append(full_value)
            fluid_scores.append(fluid_value)
            emit(f"{index:04d} full_l2={full_value:.6e} fluid_l2={fluid_value:.6e}")
            index += xx.shape[0]

    full_mean = float(np.mean(full_scores))
    fluid_mean = float(np.mean(fluid_scores))
    emit(f"full-domain relative L2: {full_mean:.6e}")
    emit(f"fluid-only relative L2: {fluid_mean:.6e}")

    scipy.io.savemat(
        output_path,
        {
            "pred": pred.cpu().numpy(),
            "u": y.cpu().numpy(),
            "mask": mask.cpu().numpy(),
            "theta": theta.cpu().numpy(),
            "t": times.cpu().numpy(),
            "full_l2_mean": np.asarray([full_mean], dtype=np.float32),
            "fluid_l2_mean": np.asarray([fluid_mean], dtype=np.float32),
        },
    )
    emit(f"Saved predictions -> {output_path}")

    if eval_metrics_path is not None:
        write_json(
            eval_metrics_path,
            {
                "full_domain_l2_mean": full_mean,
                "fluid_only_l2_mean": fluid_mean,
                "ntrain": int(args.ntrain),
                "ntest": int(args.ntest),
                "resolution": int(args.resolution),
                "record_steps": int(args.record_steps),
                "T_in": int(args.T_in),
                "T": int(args.T),
            },
        )
        emit(f"Saved evaluation metrics -> {eval_metrics_path}")


if __name__ == "__main__":
    main()
