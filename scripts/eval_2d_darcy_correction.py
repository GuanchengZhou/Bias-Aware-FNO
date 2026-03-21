#!/usr/bin/env python3
"""Evaluate Darcy structured correction runs."""

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

from darcy_correction import build_correction_model, compute_losses, darcy_flux, relative_l2
from run_artifacts import correction_artifact_paths, load_json, log_message, resolve_model_artifact, write_json
from utilities3 import MatReader, UnitGaussianNormalizer


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Darcy structured correction run.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def load_fields(path: Path, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reader = MatReader(str(path))
    coeff = reader.read_field("coeff")[:count]
    sol_coarse = reader.read_field("sol_coarse")[:count]
    sol_fine = reader.read_field("sol_fine")[:count]
    return coeff, sol_coarse, sol_fine


def infer_model_kind(model_path: Path) -> str:
    return "state_dict" if model_path.name.endswith("state_dict.pt") else "full"


def build_model_from_config(config: dict, coeff_train: torch.Tensor, sol_train_coarse: torch.Tensor):
    args = config.get("args", {})
    extra = config.get("extra", {})
    x_normalizer = UnitGaussianNormalizer(coeff_train)
    y_normalizer = UnitGaussianNormalizer(sol_train_coarse)

    backbone_modes = int(extra["backbone_modes"]) if "backbone_modes" in extra else int(args["modes"])
    backbone_width = int(extra["backbone_width"]) if "backbone_width" in extra else int(args["width"])
    coeff_resolution = int(extra["coeff_resolution"]) if "coeff_resolution" in extra else int(args.get("coeff_resolution", coeff_train.shape[-1]))

    return build_correction_model(
        coeff_resolution=coeff_resolution,
        backbone_modes=backbone_modes,
        backbone_width=backbone_width,
        correction_modes=int(args["correction_modes"]),
        correction_width=int(args["correction_width"]),
        x_mean=x_normalizer.mean,
        x_std=x_normalizer.std,
        y_mean=y_normalizer.mean,
        y_std=y_normalizer.std,
        cg_max_iter=int(args["cg_max_iter"]),
        cg_tol=float(args["cg_tol"]),
        variant=str(args.get("variant", extra.get("model_variant", "deterministic"))),
        ablation=str(args.get("ablation", extra.get("ablation", "none"))),
        disable_interface_correction=bool(args.get("disable_interface_correction", False)),
        disable_boundary_correction=bool(args.get("disable_boundary_correction", False)),
    )


def calibration_bins(pred_std_scalar: np.ndarray, actual_error_scalar: np.ndarray, n_bins: int = 5) -> tuple[list[dict], float]:
    if pred_std_scalar.size == 0:
        return [], 0.0
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(pred_std_scalar, quantiles)
    bins = []
    ece = 0.0
    total = float(pred_std_scalar.size)
    for idx in range(n_bins):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == n_bins - 1:
            mask = (pred_std_scalar >= left) & (pred_std_scalar <= right)
        else:
            mask = (pred_std_scalar >= left) & (pred_std_scalar < right)
        if not np.any(mask):
            bins.append(
                {
                    "bin": idx,
                    "count": 0,
                    "pred_std_mean": 0.0,
                    "actual_rmse_mean": 0.0,
                }
            )
            continue
        pred_mean = float(pred_std_scalar[mask].mean())
        actual_mean = float(actual_error_scalar[mask].mean())
        weight = float(mask.sum()) / total
        ece += weight * abs(pred_mean - actual_mean)
        bins.append(
            {
                "bin": idx,
                "count": int(mask.sum()),
                "pred_std_mean": pred_mean,
                "actual_rmse_mean": actual_mean,
            }
        )
    return bins, float(ece)


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size <= 1 or y.size <= 1:
        return 0.0
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    value = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(value):
        return 0.0
    return float(value)


def main() -> None:
    args = parse_args()
    if args.run_dir is None and args.model_path is None:
        raise ValueError("Either --run-dir or --model-path must be provided.")

    if args.run_dir is None and args.model_path is not None:
        candidate_run_dir = args.model_path.resolve().parent
        if (candidate_run_dir / "config.json").exists():
            args.run_dir = candidate_run_dir

    device = resolve_device(args.device)
    eval_log_path = None
    eval_metrics_path = None
    diagnostics_path = None
    predictions_path = None

    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        artifacts = correction_artifact_paths(run_dir)
        config_path = artifacts["config"]
        if not config_path.exists():
            raise FileNotFoundError(f"Correction run config not found: {config_path}")
        config = load_json(config_path)
        eval_log_path = artifacts["eval_log"]
        eval_metrics_path = artifacts["eval_metrics"]
        diagnostics_path = artifacts["correction_diagnostics"]
        predictions_path = artifacts["predictions"]
        train_path = Path(config["train_path"])
        test_path = Path(config["test_path"])
        model_path, model_kind = resolve_model_artifact(run_dir) if args.model_path is None else (args.model_path, infer_model_kind(args.model_path))
    else:
        raise ValueError("--model-path without a sibling config.json is not supported for correction evaluation.")

    def emit(message: str) -> None:
        if eval_log_path is not None:
            log_message(message, eval_log_path)
        else:
            print(message)

    coeff_train, sol_train_coarse, _ = load_fields(train_path, int(config["extra"]["ntrain"]))
    coeff_test, sol_test_coarse, sol_test_fine = load_fields(test_path, int(config["extra"]["ntest"]))

    model = build_model_from_config(config, coeff_train, sol_train_coarse).to(device)
    if model_kind == "full":
        model = torch.load(model_path, map_location=device, weights_only=False).to(device)
    else:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            emit(f"load_state_dict missing keys: {missing}")
        if unexpected:
            emit(f"load_state_dict unexpected keys: {unexpected}")
    model.eval()

    variant = str(config.get("args", {}).get("variant", config.get("extra", {}).get("model_variant", "deterministic")))
    ablation = str(config.get("args", {}).get("ablation", config.get("extra", {}).get("ablation", "none")))
    mc_samples_eval = int(config.get("extra", {}).get("mc_samples_eval", config.get("args", {}).get("mc_samples_eval", 1)))

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(coeff_test, sol_test_coarse, sol_test_fine),
        batch_size=args.batch_size,
        shuffle=False,
    )

    pred_backbone_batches = []
    pred_mean_batches = []
    pred_std_batches = []
    pred_logvar_batches = []
    residual_batches = []
    flux_error_batches = []
    correction_batches = []
    tau_x_batches = []
    tau_y_batches = []
    beta_mu_batches = []
    beta_logvar_batches = []

    coarse_scores = []
    backbone_fine_scores = []
    corrected_fine_scores = []
    flux_scores = []
    pde_scores = []
    nll_scores = []
    pred_std_scalar = []
    actual_error_scalar = []

    with torch.no_grad():
        for batch_idx, (coeff, target_coarse, target_fine) in enumerate(loader):
            coeff = coeff.to(device)
            target_coarse = target_coarse.to(device)
            target_fine = target_fine.to(device)
            outputs = model(coeff, mc_samples=mc_samples_eval if variant == "bayesian" else 1)
            losses, diagnostics = compute_losses(
                outputs,
                coeff,
                target_coarse,
                target_fine,
                variant=variant,
                disable_flux_loss=bool(config["args"].get("disable_flux_loss", False)),
                lambda_backbone=float(config["args"]["lambda_backbone"]),
                lambda_state=float(config["args"]["lambda_state"]),
                lambda_pde=float(config["args"]["lambda_pde"]),
                lambda_flux=float(config["args"]["lambda_flux"]),
                lambda_reg=float(config["args"]["lambda_reg"]),
                lambda_mask=float(config["args"]["lambda_mask"]),
                lambda_nll=float(config["args"].get("lambda_nll", 1.0)),
                lambda_kl_beta=float(config["args"].get("lambda_kl_beta", 0.0)),
                lambda_kl_var=float(config["args"].get("lambda_kl_var", 0.0)),
            )

            pred_backbone_batches.append(outputs["u_backbone"].cpu())
            pred_mean_batches.append(outputs["pred_mean"].cpu())
            pred_std_batches.append(outputs["pred_std"].cpu())
            pred_logvar_batches.append(outputs["pred_logvar"].cpu())
            residual_batches.append(outputs["residual_corrected"].cpu())
            flux_error_batches.append(diagnostics["flux_error_magnitude"].cpu())
            correction_batches.append(outputs["b_h"].cpu())
            tau_x_batches.append(outputs["tau_x"].cpu())
            tau_y_batches.append(outputs["tau_y"].cpu())

            if outputs.get("beta_mu") is not None:
                beta_mu_batches.append(outputs["beta_mu"].cpu())
            if outputs.get("beta_logvar") is not None:
                beta_logvar_batches.append(outputs["beta_logvar"].cpu())

            pred_mean = outputs["pred_mean"]
            pred_std = outputs["pred_std"]

            coarse_scores.extend(relative_l2(outputs["u_backbone"], target_coarse).cpu().numpy().tolist())
            backbone_fine_scores.extend(relative_l2(outputs["u_backbone"], target_fine).cpu().numpy().tolist())
            corrected_fine_scores.extend(relative_l2(pred_mean, target_fine).cpu().numpy().tolist())
            h = 1.0 / float(coeff.shape[-1] - 1)
            q_fine_x, q_fine_y = darcy_flux(coeff, target_fine, h)
            flux_scores.extend(
                (
                    diagnostics["flux_error_magnitude"].reshape(diagnostics["flux_error_magnitude"].shape[0], -1).norm(dim=1)
                    / torch.clamp(
                        torch.sqrt(
                            q_fine_x.square().reshape(q_fine_x.shape[0], -1).sum(dim=1)
                            + q_fine_y.square().reshape(q_fine_y.shape[0], -1).sum(dim=1)
                        ),
                        min=1e-12,
                    )
                )
                .cpu()
                .numpy()
                .tolist()
            )
            pde_scores.extend(
                outputs["residual_corrected"].reshape(outputs["residual_corrected"].shape[0], -1).norm(dim=1).cpu().numpy().tolist()
            )
            nll_scores.extend([float(losses.nll.item())] * coeff.shape[0])
            pred_std_scalar.extend(pred_std.reshape(pred_std.shape[0], -1).mean(dim=1).cpu().numpy().tolist())
            actual_error_scalar.extend(
                torch.sqrt((pred_mean - target_fine).square().reshape(pred_mean.shape[0], -1).mean(dim=1)).cpu().numpy().tolist()
            )

            emit(
                f"batch={batch_idx:04d} "
                f"variant={variant} ablation={ablation} "
                f"backbone_fine_l2={np.mean(backbone_fine_scores):.6e} "
                f"corrected_fine_l2={np.mean(corrected_fine_scores):.6e}"
            )

    pred_backbone = torch.cat(pred_backbone_batches, dim=0)
    pred_mean = torch.cat(pred_mean_batches, dim=0)
    pred_std = torch.cat(pred_std_batches, dim=0)
    pred_logvar = torch.cat(pred_logvar_batches, dim=0)
    residual_corrected = torch.cat(residual_batches, dim=0)
    flux_error_corrected = torch.cat(flux_error_batches, dim=0)
    corrections = torch.cat(correction_batches, dim=0)
    tau_x = torch.cat(tau_x_batches, dim=0)
    tau_y = torch.cat(tau_y_batches, dim=0)

    beta_mu = torch.cat(beta_mu_batches, dim=0) if beta_mu_batches else None
    beta_logvar = torch.cat(beta_logvar_batches, dim=0) if beta_logvar_batches else None

    coarse_l2_mean = float(np.mean(coarse_scores))
    backbone_fine_l2_mean = float(np.mean(backbone_fine_scores))
    corrected_fine_l2_mean = float(np.mean(corrected_fine_scores))
    flux_l2_mean = float(np.mean(flux_scores))
    pde_residual_mean = float(np.mean(pde_scores))
    nll_mean = float(np.mean(nll_scores))
    improvement_ratio = float(
        (backbone_fine_l2_mean - corrected_fine_l2_mean) / max(backbone_fine_l2_mean, 1e-12)
    )
    pred_std_scalar_np = np.asarray(pred_std_scalar)
    actual_error_scalar_np = np.asarray(actual_error_scalar)
    variance_corr = safe_correlation(pred_std_scalar_np, actual_error_scalar_np)
    bins, ece_like_error = calibration_bins(pred_std_scalar_np, actual_error_scalar_np)

    pred_payload = {
        "pred_backbone": pred_backbone.numpy().astype(np.float32),
        "pred_corrected": pred_mean.numpy().astype(np.float32),
        "target_coarse": sol_test_coarse.numpy().astype(np.float32),
        "target_fine": sol_test_fine.numpy().astype(np.float32),
        "coeff": coeff_test.numpy().astype(np.float32),
        "residual_corrected": residual_corrected.numpy().astype(np.float32),
        "flux_error_corrected": flux_error_corrected.numpy().astype(np.float32),
    }
    if variant == "bayesian":
        pred_payload.update(
            {
                "pred_mean": pred_mean.numpy().astype(np.float32),
                "pred_std": pred_std.numpy().astype(np.float32),
                "pred_logvar": pred_logvar.numpy().astype(np.float32),
            }
        )
    scipy.io.savemat(predictions_path, pred_payload)

    diag_payload = {
        "b_h": corrections.numpy().astype(np.float32),
        "tau_x": tau_x.numpy().astype(np.float32),
        "tau_y": tau_y.numpy().astype(np.float32),
        "residual_corrected": residual_corrected.numpy().astype(np.float32),
        "flux_error_corrected": flux_error_corrected.numpy().astype(np.float32),
    }
    if beta_mu is not None:
        diag_payload["beta_mu"] = beta_mu.numpy().astype(np.float32)
    if beta_logvar is not None:
        diag_payload["beta_logvar"] = beta_logvar.numpy().astype(np.float32)
    if variant == "bayesian":
        diag_payload["pred_logvar"] = pred_logvar.numpy().astype(np.float32)
    scipy.io.savemat(diagnostics_path, diag_payload)

    metrics = {
        "model_variant": variant,
        "ablation": ablation,
        "backbone_fine_l2_mean": backbone_fine_l2_mean,
        "corrected_fine_l2_mean": corrected_fine_l2_mean,
        "coarse_l2_mean": coarse_l2_mean,
        "backbone_coarse_l2_mean": coarse_l2_mean,
        "flux_l2_mean": flux_l2_mean,
        "pde_residual_mean": pde_residual_mean,
        "correction_improvement_ratio": improvement_ratio,
        "correction_train_samples": int(config["extra"]["correction_train_samples"]),
        "nll_mean": nll_mean,
        "calibration_bins": bins,
        "ece_like_error": float(ece_like_error),
        "variance_error_correlation": variance_corr,
    }
    write_json(eval_metrics_path, metrics)

    emit(f"Saved predictions -> {predictions_path}")
    emit(f"Saved diagnostics -> {diagnostics_path}")
    emit(f"Saved evaluation metrics -> {eval_metrics_path}")
    emit(
        f"summary coarse_l2={coarse_l2_mean:.6e} "
        f"backbone_fine_l2={backbone_fine_l2_mean:.6e} "
        f"corrected_fine_l2={corrected_fine_l2_mean:.6e} "
        f"nll={nll_mean:.6e}"
    )


if __name__ == "__main__":
    main()
