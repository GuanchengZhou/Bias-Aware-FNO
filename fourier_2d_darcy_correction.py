#!/usr/bin/env python3
"""Train Stage 2/3 Darcy structured correction from a pretrained Stage 1 backbone."""

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
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from Adam import Adam
from darcy_correction import build_correction_model, compute_losses
from run_artifacts import (
    append_csv_row,
    build_run_config,
    correction_artifact_paths,
    ensure_run_dir,
    load_json,
    log_message,
    resolve_model_artifact,
    write_json,
)
from utilities3 import MatReader, UnitGaussianNormalizer, count_params


REPO_ROOT = Path(__file__).resolve().parent
MPLCONFIGDIR = REPO_ROOT / "tmp" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt


def parse_bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_fields(path: Path, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reader = MatReader(str(path))
    coeff = reader.read_field("coeff")[:count]
    sol_coarse = reader.read_field("sol_coarse")[:count]
    sol_fine = reader.read_field("sol_fine")[:count]
    return coeff, sol_coarse, sol_fine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 2/3 Darcy correction from a Stage 1 run.")
    parser.add_argument("--backbone-run-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["deterministic", "bayesian"], default="deterministic")
    parser.add_argument("--ablation", choices=["none", "direct-bias", "direct-flux"], default="none")
    parser.add_argument("--correction-train-samples", type=int, default=100)
    parser.add_argument("--stage2-epochs", type=int, default=50)
    parser.add_argument("--stage3-epochs", type=int, default=50)
    parser.add_argument("--correction-modes", type=int, default=8)
    parser.add_argument("--correction-width", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stage2-learning-rate", type=float, default=1)
    parser.add_argument("--stage3-learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cg-max-iter", type=int, default=30)
    parser.add_argument("--cg-tol", type=float, default=1e-6)
    parser.add_argument("--mc-samples-train", type=int, default=2)
    parser.add_argument("--mc-samples-eval", type=int, default=4)
    parser.add_argument("--freeze-backbone-stage2", type=parse_bool, default=True)
    parser.add_argument("--disable-interface-correction", action="store_true")
    parser.add_argument("--disable-boundary-correction", action="store_true")
    parser.add_argument("--disable-flux-loss", action="store_true")
    parser.add_argument("--lambda-backbone", type=float, default=0.5)
    parser.add_argument("--lambda-state", type=float, default=1.0)
    parser.add_argument("--lambda-pde", type=float, default=0.1)
    parser.add_argument("--lambda-flux", type=float, default=0.2)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--lambda-mask", type=float, default=1e-3)
    parser.add_argument("--lambda-nll", type=float, default=1.0)
    parser.add_argument("--lambda-kl-beta", type=float, default=1e-4)
    parser.add_argument("--lambda-kl-var", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-sample-plot", action="store_true")
    return parser.parse_args()


def load_backbone_state_dict(backbone_run_dir: Path) -> tuple[dict, dict, Path]:
    config = load_json(correction_artifact_paths(backbone_run_dir)["config"])
    model_path, model_kind = resolve_model_artifact(backbone_run_dir)
    if model_kind == "full":
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = model.state_dict()
    else:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    return config, state_dict, model_path


def build_experiment_name(args: argparse.Namespace, backbone_run_dir: Path) -> str:
    parts = [f"{backbone_run_dir.name}_correction"]
    if args.variant != "deterministic":
        parts.append(args.variant)
    if args.ablation != "none":
        parts.append(args.ablation)
    if args.disable_interface_correction:
        parts.append("no-interface")
    if args.disable_boundary_correction:
        parts.append("no-boundary")
    if args.disable_flux_loss:
        parts.append("no-flux-loss")
    return "_".join(parts)


def save_sample_plot(
    path: Path,
    coeff: torch.Tensor,
    backbone: torch.Tensor,
    corrected: torch.Tensor,
    target_fine: torch.Tensor,
    correction: torch.Tensor,
    pred_std: torch.Tensor | None = None,
) -> None:
    panels = [
        (coeff, "Coefficient", "viridis"),
        (backbone, "Backbone", "RdBu_r"),
        (corrected, "Corrected", "RdBu_r"),
        (target_fine, "Fine Target", "RdBu_r"),
        (target_fine - corrected, "Fine - Corrected", "RdBu_r"),
        (correction, "Correction b_h", "RdBu_r"),
    ]
    if pred_std is not None:
        panels.append((pred_std, "Predictive Std", "magma"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (field, title, cmap) in zip(axes, panels):
        im = ax.imshow(field.cpu().numpy(), cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def set_backbone_requires_grad(model, enabled: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = enabled


def collect_trainable_parameters(model) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def metric_fieldnames() -> list[str]:
    return [
        "epoch",
        "time_sec",
        "loss_total",
        "loss_backbone_coarse",
        "loss_state_fine",
        "loss_pde",
        "loss_flux",
        "loss_reg",
        "loss_mask",
        "loss_nll",
        "loss_kl_beta",
        "loss_kl_var",
        "fine_l2_corrected",
        "fine_l2_backbone",
    ]


def aggregate_loss_row(losses, diagnostics, batch_size: int) -> dict[str, float]:
    return {
        "loss_total": losses.total.item() * batch_size,
        "loss_backbone_coarse": losses.backbone_coarse.item() * batch_size,
        "loss_state_fine": losses.state_fine.item() * batch_size,
        "loss_pde": losses.pde.item() * batch_size,
        "loss_flux": losses.flux.item() * batch_size,
        "loss_reg": losses.reg.item() * batch_size,
        "loss_mask": losses.mask.item() * batch_size,
        "loss_nll": losses.nll.item() * batch_size,
        "loss_kl_beta": losses.kl_beta.item() * batch_size,
        "loss_kl_var": losses.kl_var.item() * batch_size,
        "fine_l2_corrected": diagnostics["fine_l2_corrected"].sum().item(),
        "fine_l2_backbone": diagnostics["fine_l2_backbone"].sum().item(),
    }


def evaluate_model(
    model,
    loader,
    device: torch.device,
    *,
    variant: str,
    mc_samples: int,
    disable_flux_loss: bool,
    lambda_backbone: float,
    lambda_state: float,
    lambda_pde: float,
    lambda_flux: float,
    lambda_reg: float,
    lambda_mask: float,
    lambda_nll: float,
    lambda_kl_beta: float,
    lambda_kl_var: float,
):
    model.eval()
    totals = {name: 0.0 for name in metric_fieldnames() if name not in {"epoch", "time_sec"}}
    count = 0
    last_sample = None

    with torch.no_grad():
        for coeff, target_coarse, target_fine in loader:
            coeff = coeff.to(device)
            target_coarse = target_coarse.to(device)
            target_fine = target_fine.to(device)

            outputs = model(coeff, mc_samples=mc_samples)
            losses, diagnostics = compute_losses(
                outputs,
                coeff,
                target_coarse,
                target_fine,
                variant=variant,
                disable_flux_loss=disable_flux_loss,
                lambda_backbone=lambda_backbone,
                lambda_state=lambda_state,
                lambda_pde=lambda_pde,
                lambda_flux=lambda_flux,
                lambda_reg=lambda_reg,
                lambda_mask=lambda_mask,
                lambda_nll=lambda_nll,
                lambda_kl_beta=lambda_kl_beta,
                lambda_kl_var=lambda_kl_var,
            )

            batch_size = coeff.shape[0]
            for key, value in aggregate_loss_row(losses, diagnostics, batch_size).items():
                totals[key] += value
            count += batch_size

            last_sample = {
                "coeff": coeff[0].detach().cpu(),
                "u_backbone": outputs["u_backbone"][0].detach().cpu(),
                "u_corrected": outputs["pred_mean"][0].detach().cpu(),
                "target_fine": target_fine[0].detach().cpu(),
                "b_h": outputs["b_h"][0].detach().cpu(),
                "pred_std": outputs["pred_std"][0].detach().cpu(),
            }

    if count == 0:
        raise RuntimeError("Empty evaluation loader.")

    return {key: value / count for key, value in totals.items()}, last_sample


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    backbone_run_dir = args.backbone_run_dir.resolve()
    backbone_artifacts = correction_artifact_paths(backbone_run_dir)
    if not backbone_artifacts["config"].exists():
        raise FileNotFoundError(f"Baseline Stage 1 run config not found: {backbone_artifacts['config']}")

    backbone_config, backbone_state_dict, backbone_model_path = load_backbone_state_dict(backbone_run_dir)
    backbone_args = backbone_config.get("args", {})
    train_path = Path(backbone_config["train_path"])
    test_path = Path(backbone_config["test_path"])
    ntrain = int(backbone_args["ntrain"])
    ntest = int(backbone_args["ntest"])
    coeff_resolution = int(backbone_args["coeff_resolution"])
    coarse_n = int(backbone_args["coarse_n"])
    fine_n = int(backbone_args["fine_n"])
    backbone_modes = int(backbone_args["modes"])
    backbone_width = int(backbone_args["width"])

    coeff_train_full, sol_train_coarse_full, sol_train_fine_full = load_fields(train_path, ntrain)
    coeff_test, sol_test_coarse, sol_test_fine = load_fields(test_path, ntest)

    x_normalizer = UnitGaussianNormalizer(coeff_train_full)
    y_normalizer = UnitGaussianNormalizer(sol_train_coarse_full)

    correction_train_samples = min(int(args.correction_train_samples), int(coeff_train_full.shape[0]))
    coeff_train = coeff_train_full[:correction_train_samples]
    sol_train_coarse = sol_train_coarse_full[:correction_train_samples]
    sol_train_fine = sol_train_fine_full[:correction_train_samples]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(coeff_train, sol_train_coarse, sol_train_fine),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(coeff_test, sol_test_coarse, sol_test_fine),
        batch_size=args.batch_size,
        shuffle=False,
    )

    experiment_name = build_experiment_name(args, backbone_run_dir)
    run_dir = ensure_run_dir(REPO_ROOT, experiment_name)
    artifacts = correction_artifact_paths(run_dir)
    config = build_run_config(
        task="darcy_correction",
        script_name=Path(__file__).name,
        experiment_name=run_dir.name,
        run_dir=run_dir,
        train_path=train_path,
        test_path=test_path,
        args=args,
        extra={
            "model_variant": args.variant,
            "ablation": args.ablation,
            "backbone_run_dir": str(backbone_run_dir),
            "backbone_model_path": str(backbone_model_path),
            "backbone_experiment_name": backbone_run_dir.name,
            "baseline_task": backbone_config.get("task"),
            "ntrain": ntrain,
            "ntest": ntest,
            "coeff_resolution": coeff_resolution,
            "coarse_n": coarse_n,
            "fine_n": fine_n,
            "backbone_modes": backbone_modes,
            "backbone_width": backbone_width,
            "correction_train_samples": correction_train_samples,
            "loss_weights": {
                "lambda_backbone": args.lambda_backbone,
                "lambda_state": args.lambda_state,
                "lambda_pde": args.lambda_pde,
                "lambda_flux": args.lambda_flux,
                "lambda_reg": args.lambda_reg,
                "lambda_mask": args.lambda_mask,
                "lambda_nll": args.lambda_nll,
                "lambda_kl_beta": args.lambda_kl_beta,
                "lambda_kl_var": args.lambda_kl_var,
            },
            "mc_samples_train": args.mc_samples_train,
            "mc_samples_eval": args.mc_samples_eval,
        },
    )
    config["artifacts"].update(
        {
            "stage2_metrics": str(artifacts["stage2_metrics"]),
            "stage3_metrics": str(artifacts["stage3_metrics"]),
            "correction_diagnostics": str(artifacts["correction_diagnostics"]),
        }
    )
    write_json(artifacts["config"], config)

    device = resolve_device(args.device)
    model = build_correction_model(
        coeff_resolution=coeff_resolution,
        backbone_modes=backbone_modes,
        backbone_width=backbone_width,
        correction_modes=args.correction_modes,
        correction_width=args.correction_width,
        x_mean=x_normalizer.mean,
        x_std=x_normalizer.std,
        y_mean=y_normalizer.mean,
        y_std=y_normalizer.std,
        cg_max_iter=args.cg_max_iter,
        cg_tol=args.cg_tol,
        variant=args.variant,
        ablation=args.ablation,
        disable_interface_correction=args.disable_interface_correction,
        disable_boundary_correction=args.disable_boundary_correction,
    ).to(device)
    model.backbone.model.load_state_dict(backbone_state_dict)

    log_message(f"run_dir {run_dir}", artifacts["train_log"], tqdm_module=tqdm)
    log_message(f"baseline_run_dir {backbone_run_dir}", artifacts["train_log"], tqdm_module=tqdm)
    log_message(f"baseline_model_path {backbone_model_path}", artifacts["train_log"], tqdm_module=tqdm)
    log_message(
        f"variant={args.variant} ablation={args.ablation} "
        f"train_subset {correction_train_samples}/{coeff_train_full.shape[0]} "
        f"train_coeff {tuple(coeff_train.shape)} test_coeff {tuple(coeff_test.shape)}",
        artifacts["train_log"],
        tqdm_module=tqdm,
    )
    log_message(f"params {count_params(model)}", artifacts["train_log"], tqdm_module=tqdm)

    last_sample = None

    def run_stage(stage_name: str, epochs: int, learning_rate: float, train_backbone: bool, csv_path: Path) -> None:
        nonlocal last_sample
        if epochs <= 0:
            log_message(f"Skipping {stage_name}: epochs={epochs}", artifacts["train_log"], tqdm_module=tqdm)
            return

        set_backbone_requires_grad(model, train_backbone)
        optimizer = Adam(collect_trainable_parameters(model), lr=learning_rate, weight_decay=args.weight_decay)

        epoch_iterator = range(epochs)
        if tqdm is not None:
            epoch_iterator = tqdm(
                epoch_iterator,
                desc=f"{stage_name} Epochs",
                dynamic_ncols=True,
                position=0,
            )

        for epoch in epoch_iterator:
            t1 = default_timer()
            model.train()
            running = {name: 0.0 for name in metric_fieldnames() if name not in {"epoch", "time_sec"}}
            train_count = 0

            batch_iterator = train_loader
            if tqdm is not None:
                batch_iterator = tqdm(
                    train_loader,
                    desc=f"{stage_name} Train {epoch + 1:04d}/{epochs:04d}",
                    total=len(train_loader),
                    dynamic_ncols=True,
                    position=1,
                    leave=False,
                )

            for coeff, target_coarse, target_fine in batch_iterator:
                coeff = coeff.to(device)
                target_coarse = target_coarse.to(device)
                target_fine = target_fine.to(device)

                optimizer.zero_grad()
                outputs = model(coeff, mc_samples=args.mc_samples_train if args.variant == "bayesian" else 1)
                losses, diagnostics = compute_losses(
                    outputs,
                    coeff,
                    target_coarse,
                    target_fine,
                    variant=args.variant,
                    disable_flux_loss=args.disable_flux_loss,
                    lambda_backbone=args.lambda_backbone,
                    lambda_state=args.lambda_state,
                    lambda_pde=args.lambda_pde,
                    lambda_flux=args.lambda_flux,
                    lambda_reg=args.lambda_reg,
                    lambda_mask=args.lambda_mask,
                    lambda_nll=args.lambda_nll,
                    lambda_kl_beta=args.lambda_kl_beta,
                    lambda_kl_var=args.lambda_kl_var,
                )
                losses.total.backward()
                optimizer.step()

                batch_size = coeff.shape[0]
                for key, value in aggregate_loss_row(losses, diagnostics, batch_size).items():
                    running[key] += value
                train_count += batch_size

                if tqdm is not None:
                    batch_iterator.set_postfix(
                        total=f"{running['loss_total'] / train_count:.3e}",
                        fine=f"{diagnostics['fine_l2_corrected'].mean().item():.3e}",
                    )

            t2 = default_timer()
            eval_metrics, last_sample = evaluate_model(
                model,
                test_loader,
                device,
                variant=args.variant,
                mc_samples=args.mc_samples_eval if args.variant == "bayesian" else 1,
                disable_flux_loss=args.disable_flux_loss,
                lambda_backbone=args.lambda_backbone,
                lambda_state=args.lambda_state,
                lambda_pde=args.lambda_pde,
                lambda_flux=args.lambda_flux,
                lambda_reg=args.lambda_reg,
                lambda_mask=args.lambda_mask,
                lambda_nll=args.lambda_nll,
                lambda_kl_beta=args.lambda_kl_beta,
                lambda_kl_var=args.lambda_kl_var,
            )

            row = {
                "epoch": epoch,
                "time_sec": t2 - t1,
                "loss_total": running["loss_total"] / train_count,
                "loss_backbone_coarse": running["loss_backbone_coarse"] / train_count,
                "loss_state_fine": running["loss_state_fine"] / train_count,
                "loss_pde": running["loss_pde"] / train_count,
                "loss_flux": running["loss_flux"] / train_count,
                "loss_reg": running["loss_reg"] / train_count,
                "loss_mask": running["loss_mask"] / train_count,
                "loss_nll": running["loss_nll"] / train_count,
                "loss_kl_beta": running["loss_kl_beta"] / train_count,
                "loss_kl_var": running["loss_kl_var"] / train_count,
                "fine_l2_corrected": eval_metrics["fine_l2_corrected"],
                "fine_l2_backbone": eval_metrics["fine_l2_backbone"],
            }
            append_csv_row(csv_path, metric_fieldnames(), row)
            summary = (
                f"{stage_name} epoch={epoch:04d} time={t2 - t1:.4f} "
                f"loss_total={row['loss_total']:.6e} "
                f"loss_backbone={row['loss_backbone_coarse']:.6e} "
                f"loss_state={row['loss_state_fine']:.6e} "
                f"loss_pde={row['loss_pde']:.6e} "
                f"loss_flux={row['loss_flux']:.6e} "
                f"loss_nll={row['loss_nll']:.6e} "
                f"loss_kl_beta={row['loss_kl_beta']:.6e} "
                f"loss_kl_var={row['loss_kl_var']:.6e} "
                f"fine_l2_corrected={row['fine_l2_corrected']:.6e} "
                f"fine_l2_backbone={row['fine_l2_backbone']:.6e}"
            )
            log_message(summary, artifacts["train_log"], tqdm_module=tqdm)

            if tqdm is not None:
                epoch_iterator.set_postfix(
                    total=f"{row['loss_total']:.3e}",
                    fine_corr=f"{row['fine_l2_corrected']:.3e}",
                    fine_backbone=f"{row['fine_l2_backbone']:.3e}",
                )

    run_stage(
        "Stage2",
        args.stage2_epochs,
        args.stage2_learning_rate,
        not args.freeze_backbone_stage2,
        artifacts["stage2_metrics"],
    )
    run_stage("Stage3", args.stage3_epochs, args.stage3_learning_rate, True, artifacts["stage3_metrics"])

    torch.save(model, artifacts["model"])
    torch.save(model.state_dict(), artifacts["model_state_dict"])
    log_message(f"Saved model -> {artifacts['model']}", artifacts["train_log"], tqdm_module=tqdm)
    log_message(
        f"Saved model_state_dict -> {artifacts['model_state_dict']}",
        artifacts["train_log"],
        tqdm_module=tqdm,
    )

    if args.save_sample_plot and last_sample is not None:
        plot_std = last_sample["pred_std"] if args.variant == "bayesian" else None
        save_sample_plot(
            artifacts["sample"],
            last_sample["coeff"],
            last_sample["u_backbone"],
            last_sample["u_corrected"],
            last_sample["target_fine"],
            last_sample["b_h"],
            pred_std=plot_std,
        )
        log_message(f"Saved sample plot -> {artifacts['sample']}", artifacts["train_log"], tqdm_module=tqdm)

    scipy.io.savemat(
        artifacts["train_summary"],
        {
            "train_coeff_shape": np.asarray(coeff_train.shape, dtype=np.int64),
            "test_coeff_shape": np.asarray(coeff_test.shape, dtype=np.int64),
            "coeff_resolution": np.asarray([coeff_resolution], dtype=np.int64),
            "correction_train_samples": np.asarray([correction_train_samples], dtype=np.int64),
            "stage2_epochs": np.asarray([args.stage2_epochs], dtype=np.int64),
            "stage3_epochs": np.asarray([args.stage3_epochs], dtype=np.int64),
            "variant": np.asarray([args.variant], dtype=object),
            "ablation": np.asarray([args.ablation], dtype=object),
        },
    )
    log_message(f"Saved train summary -> {artifacts['train_summary']}", artifacts["train_log"], tqdm_module=tqdm)

    eval_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_2d_darcy_correction.py"),
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
