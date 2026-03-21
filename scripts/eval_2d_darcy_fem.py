#!/usr/bin/env python3
"""Evaluate a Darcy FNO model against coarse and fine FEM targets on the shared 421 grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fourier_2d_darcy_fem import (
    FNO2d,
    SpectralConv2d,
    build_inputs,
    default_dataset_path,
    format_experiment_name,
    resolve_device,
)
from run_artifacts import load_json, log_message, resolve_model_artifact, standard_artifact_paths, write_json
from utilities3 import LpLoss, MatReader, UnitGaussianNormalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Darcy FNO predictions on coarse/fine FEM targets.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--test-path", type=Path, default=None)
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=100)
    parser.add_argument("--coeff-resolution", type=int, default=256)
    parser.add_argument("--coarse-n", type=int, default=150)
    parser.add_argument("--fine-n", type=int, default=300)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-name", type=str, default=None)
    return parser.parse_args()


def load_fields(path: Path, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reader = MatReader(str(path))
    coeff = reader.read_field("coeff")[:count]
    sol_coarse = reader.read_field("sol_coarse")[:count]
    sol_fine = reader.read_field("sol_fine")[:count]
    return coeff, sol_coarse, sol_fine


def hydrate_args_from_run_config(args: argparse.Namespace, config: dict) -> tuple[Path, Path]:
    stored_args = config.get("args", {})
    for key in ["ntrain", "ntest", "coeff_resolution", "coarse_n", "fine_n", "modes", "width", "epochs"]:
        if key in stored_args:
            setattr(args, key, stored_args[key])
    train_path = args.train_path or Path(config["train_path"])
    test_path = args.test_path or Path(config["test_path"])
    return train_path, test_path


def load_model(model_path: Path, model_kind: str, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if model_kind == "full":
        model = torch.load(model_path, map_location=device, weights_only=False)
        return model.to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model = FNO2d(args.modes, args.modes, args.width).to(device)
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
        train_path, test_path = hydrate_args_from_run_config(args, config)
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
        if args.model_path is None:
            exp_name = format_experiment_name(train_path, args.ntrain, args.epochs, args.modes, args.width)
            model_path = REPO_ROOT / "model" / exp_name
        else:
            model_path = args.model_path
        model_kind = infer_model_kind(model_path)
        output_name = args.output_name or f"{model_path.name}_darcy_eval"
        output_path = REPO_ROOT / "pred" / f"{output_name}.mat"
        output_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(message: str) -> None:
        if eval_log_path is not None:
            log_message(message, eval_log_path)
        else:
            print(message)

    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    emit(f"run_dir {run_dir}" if run_dir is not None else "run_dir <legacy>")
    emit(f"train_path {train_path}")
    emit(f"test_path {test_path}")
    emit(f"model_path {model_path}")

    coeff_train, y_train_coarse, _ = load_fields(train_path, args.ntrain)
    coeff_test, y_test_coarse, y_test_fine = load_fields(test_path, args.ntest)

    x_normalizer = UnitGaussianNormalizer(coeff_train)
    coeff_test_norm = x_normalizer.encode(coeff_test)
    y_normalizer = UnitGaussianNormalizer(y_train_coarse)
    if device.type == "cuda":
        y_normalizer.cuda()

    x_test = build_inputs(coeff_test_norm)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, coeff_test, y_test_coarse, y_test_fine),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = load_model(model_path, model_kind, args, device)
    model.eval()
    loss_fn = LpLoss(size_average=False)

    pred_batches = []
    coarse_scores = []
    fine_scores = []

    with torch.no_grad():
        for batch_idx, (xx, coeff_batch, yy_coarse, yy_fine) in enumerate(loader):
            xx = xx.to(device)
            yy_coarse = yy_coarse.to(device)
            yy_fine = yy_fine.to(device)
            pred = y_normalizer.decode(model(xx))
            pred_batches.append(pred.cpu())

            coarse_value = loss_fn(pred.reshape(pred.shape[0], -1), yy_coarse.reshape(yy_coarse.shape[0], -1)).item()
            fine_value = loss_fn(pred.reshape(pred.shape[0], -1), yy_fine.reshape(yy_fine.shape[0], -1)).item()
            coarse_scores.append(coarse_value)
            fine_scores.append(fine_value)
            emit(
                f"batch={batch_idx:04d} "
                f"coarse_l2={coarse_value:.6e} fine_l2={fine_value:.6e}"
            )

    pred = torch.cat(pred_batches, dim=0)
    coarse_mean = float(np.mean(coarse_scores))
    fine_mean = float(np.mean(fine_scores))

    emit(f"coarse reference relative L2: {coarse_mean:.6e}")
    emit(f"fine relative L2: {fine_mean:.6e}")

    scipy.io.savemat(
        output_path,
        {
            "pred": pred.numpy().astype(np.float32),
            "target_coarse": y_test_coarse.numpy().astype(np.float32),
            "target_fine": y_test_fine.numpy().astype(np.float32),
            "coeff": coeff_test.numpy().astype(np.float32),
            "coarse_l2_mean": np.asarray([coarse_mean], dtype=np.float32),
            "fine_l2_mean": np.asarray([fine_mean], dtype=np.float32),
        },
    )
    emit(f"Saved predictions -> {output_path}")

    if eval_metrics_path is not None:
        write_json(
            eval_metrics_path,
            {
                "coarse_l2_mean": coarse_mean,
                "fine_l2_mean": fine_mean,
                "ntrain": int(args.ntrain),
                "ntest": int(args.ntest),
                "coeff_resolution": int(args.coeff_resolution),
                "coarse_n": int(args.coarse_n),
                "fine_n": int(args.fine_n),
            },
        )
        emit(f"Saved evaluation metrics -> {eval_metrics_path}")


if __name__ == "__main__":
    main()
