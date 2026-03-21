#!/usr/bin/env python3
"""Shared helpers for per-run experiment artifacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def ensure_run_dir(repo_root: Path, experiment_name: str) -> Path:
    runs_root = repo_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = runs_root / experiment_name
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    index = 2
    while True:
        candidate = runs_root / f"{experiment_name}_run{index}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        index += 1


def standard_artifact_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "config": run_dir / "config.json",
        "model": run_dir / "model.pt",
        "model_state_dict": run_dir / "model_state_dict.pt",
        "train_log": run_dir / "train.log",
        "train_metrics": run_dir / "train_metrics.csv",
        "test_metrics": run_dir / "test_metrics.csv",
        "eval_log": run_dir / "eval.log",
        "eval_metrics": run_dir / "eval_metrics.json",
        "predictions": run_dir / "predictions.mat",
        "sample": run_dir / "sample.png",
        "train_summary": run_dir / "train_summary.mat",
    }


def _jsonify(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def namespace_to_dict(args) -> dict:
    return {key: _jsonify(value) for key, value in vars(args).items()}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def append_csv_row(path: Path, fieldnames: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: _jsonify(row.get(name)) for name in fieldnames})


def log_message(message: str, log_path: Path, tqdm_module=None) -> None:
    if tqdm_module is not None:
        tqdm_module.write(message)
    else:
        print(message)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def build_run_config(
    *,
    task: str,
    script_name: str,
    experiment_name: str,
    run_dir: Path,
    train_path: Path | None,
    test_path: Path | None,
    args,
    extra: dict | None = None,
) -> dict:
    artifacts = {name: str(path) for name, path in standard_artifact_paths(run_dir).items()}
    payload = {
        "task": task,
        "script_name": script_name,
        "experiment_name": experiment_name,
        "run_dir": str(run_dir),
        "created_at": datetime.now().astimezone().isoformat(),
        "train_path": str(train_path) if train_path is not None else None,
        "test_path": str(test_path) if test_path is not None else None,
        "artifacts": artifacts,
        "args": namespace_to_dict(args),
    }
    if extra:
        payload["extra"] = _jsonify(extra)
    return payload


def resolve_model_artifact(run_dir: Path) -> tuple[Path, str]:
    paths = standard_artifact_paths(run_dir)
    if paths["model_state_dict"].exists():
        return paths["model_state_dict"], "state_dict"
    if paths["model"].exists():
        return paths["model"], "full"
    raise FileNotFoundError(f"No model artifact found in {run_dir}")
