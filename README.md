# Bias_Aware_FNO

FNO-style training and evaluation scripts for random five-hole Navier-Stokes data.

The repo now also includes an FNO-style Darcy Flow pipeline with paired coarse/fine FEniCSx FEM datasets.

Data generation now uses `gmsh + dolfinx + PETSc` in `fenicsx-env`.
Training and evaluation still use the original FNO-style PyTorch scripts in `torch_310`.

Each training run now writes a self-contained result bundle to:
```bash
runs/<experiment_name>/
```

Typical contents include:
- `config.json`
- `model.pt`
- `model_state_dict.pt`
- `train.log`
- `train_metrics.csv`
- `test_metrics.csv`
- `eval.log`
- `eval_metrics.json`
- `predictions.mat`
- `sample.png`
- `train_summary.mat`

PyTorch environment:
```bash
/Users/zhougc/miniconda3/envs/torch_310/bin/python
```

FEniCSx environment:
```bash
/Users/zhougc/miniconda3/envs/fenicsx-env/bin/python
```

## Layout
- `Adam.py`, `utilities3.py`: utilities kept in the original `fourier-neural-operator` style.
- `data_generation/navier_stokes_5holes/ns_2d_fenicsx.py`: default FEniCSx dataset generator.
- `data_generation/navier_stokes_5holes/ns_2d_brinkman.py`: legacy spectral baseline.
- `fourier_2d_time_5holes.py`: recurrent one-step FNO training script.
- `scripts/eval_2d_time_5holes.py`: rollout evaluation script.
- `data_generation/darcy/darcy_fem_fenicsx.py`: paired coarse/fine Darcy FEM generator.
- `fourier_2d_darcy_fem.py`: static FNO-Darcy training script with coarse supervision and fine testing.
- `scripts/eval_2d_darcy_fem.py`: Darcy evaluation against coarse and fine targets on the shared grid.
- `darcy_correction.py`: structured flux correction model on the shared Darcy grid.
- `fourier_2d_darcy_correction.py`: Stage 2 / Stage 3 correction training on top of a Stage 1 run.
- `scripts/eval_2d_darcy_correction.py`: correction-run evaluation with backbone vs corrected diagnostics.

## Data Generation
```bash
cd /Users/zhougc/Desktop/IID/LRTOR_project/Bias_Aware_FNO
/Users/zhougc/miniconda3/envs/fenicsx-env/bin/python data_generation/navier_stokes_5holes/ns_2d_fenicsx.py
```

Large datasets automatically fall back to an HDF5-backed `.mat` file when classic MAT v5 would exceed the 4 GB array limit. The existing training scripts and visualization notebook can read both formats.

The FEniCSx generator now uses a Brinkman-aligned setup for:
- Gaussian random field initial vorticity,
- low-frequency forcing exported as `f`,
- smoothed obstacle mask exported as `chi_smooth`.

Useful controls:
- `--initial-vorticity-scale`: make the initial condition more or less aggressive.
- `--grf-alpha`, `--grf-tau`: change the GRF smoothness and correlation scale.
- `--mesh-density`: directly control finite-element mesh density; larger values give finer gmsh meshes and override `--mesh-size-min/--mesh-size-max`.

Smoke test:
```bash
/Users/zhougc/miniconda3/envs/fenicsx-env/bin/python data_generation/navier_stokes_5holes/ns_2d_fenicsx.py \
  --grid-resolution 64 \
  --n-train 2 \
  --n-test 1 \
  --record-steps 10 \
  --final-time 1.0 \
  --dt 0.1 \
  --mesh-density 10
```

Legacy spectral baseline:
```bash
/Users/zhougc/miniconda3/envs/torch_310/bin/python data_generation/navier_stokes_5holes/ns_2d_brinkman.py
```

## Darcy Data Generation
```bash
/Users/zhougc/miniconda3/envs/fenicsx-env/bin/python data_generation/darcy/darcy_fem_fenicsx.py
```

Default behavior:
- high-resolution coefficient field `coeff` follows the FNO-Darcy GRF threshold construction,
- coarse/fine FEM solves use `P1` on `create_unit_square` meshes,
- the default FEM meshes are `256 x 256` and `512 x 512`,
- both FEM solutions are evaluated exactly onto the same `421 x 421` reference grid,
- train targets are `sol_coarse`,
- test targets are `sol_fine`.

Useful controls:
- `--coeff-resolution 421` sets the shared reference grid used for `coeff`, `sol_coarse`, and `sol_fine`.
- `--coarse-n 256 --fine-n 512` sets the FEM mesh densities.
- `--coeff-low 4 --coeff-high 12` changes the two-value medium.

Smoke test:
```bash
/Users/zhougc/miniconda3/envs/fenicsx-env/bin/python data_generation/darcy/darcy_fem_fenicsx.py \
  --coeff-resolution 129 \
  --coarse-n 32 \
  --fine-n 64 \
  --n-train 4 \
  --n-test 2
```

## Training
```bash
/Users/zhougc/miniconda3/envs/torch_310/bin/python fourier_2d_time_5holes.py
```

Smoke test:
```bash
/Users/zhougc/miniconda3/envs/torch_310/bin/python fourier_2d_time_5holes.py \
  --ntrain 8 \
  --ntest 4 \
  --resolution 64 \
  --record-steps 10 \
  --sub 1 \
  --T-in 4 \
  --T 4 \
  --epochs 2 \
  --batch-size 2 \
  --device cpu \
  --save-sample-plot
```

Darcy training:
```bash
/Users/zhougc/miniconda3/envs/torch_310/bin/python fourier_2d_darcy_fem.py
```

Darcy smoke test:
```bash
/Users/zhougc/miniconda3/envs/torch_310/bin/python fourier_2d_darcy_fem.py \
  --ntrain 4 \
  --ntest 2 \
  --coeff-resolution 129 \
  --coarse-n 32 \
  --fine-n 64 \
  --epochs 2 \
  --batch-size 1 \
  --device cpu \
  --save-sample-plot
```

Darcy training uses:
- `x_train = coeff`
- `y_train = sol_coarse`
- `x_test = coeff`
- `y_test = sol_fine`

There is no `sub` downsampling in the Darcy pipeline anymore.
After training finishes, the script automatically runs evaluation and stores everything in the new run directory.

Darcy correction training:
```bash
/Users/zhougc/miniconda3/envs/torch_310/bin/python fourier_2d_darcy_correction.py \
  --backbone-run-dir runs/<stage1_experiment_name>
```

Darcy correction smoke test:
```bash
/Users/zhougc/miniconda3/envs/torch_310/bin/python fourier_2d_darcy_correction.py \
  --backbone-run-dir runs/darcy_fem_r129_C32_F64_N4_train_fourier_2d_darcy_N4_ep1_m12_w32 \
  --stage2-epochs 1 \
  --stage3-epochs 1 \
  --batch-size 1 \
  --device cpu \
  --save-sample-plot
```

Darcy correction uses:
- Stage 1 baseline backbone from an existing run directory
- Stage 2 and Stage 3 only on the first `100` training samples, or all samples if fewer than `100`
- coarse supervision for the backbone term
- fine supervision for the corrected output

## Evaluation
```bash
/Users/zhougc/miniconda3/envs/torch_310/bin/python scripts/eval_2d_time_5holes.py
```

Run-directory evaluation:
```bash
/Users/zhougc/miniconda3/envs/torch_310/bin/python scripts/eval_2d_time_5holes.py \
  --run-dir runs/<experiment_name>
```

Smoke test:
```bash
/Users/zhougc/miniconda3/envs/torch_310/bin/python scripts/eval_2d_time_5holes.py \
  --ntrain 8 \
  --ntest 4 \
  --resolution 64 \
  --record-steps 10 \
  --sub 1 \
  --T-in 4 \
  --T 4 \
  --epochs 2 \
  --device cpu
```

Darcy evaluation:
```bash
/Users/zhougc/miniconda3/envs/torch_310/bin/python scripts/eval_2d_darcy_fem.py
```

Run-directory evaluation:
```bash
/Users/zhougc/miniconda3/envs/torch_310/bin/python scripts/eval_2d_darcy_fem.py \
  --run-dir runs/<experiment_name>
```

Darcy smoke test:
```bash
/Users/zhougc/miniconda3/envs/torch_310/bin/python scripts/eval_2d_darcy_fem.py \
  --ntrain 4 \
  --ntest 2 \
  --coeff-resolution 129 \
  --coarse-n 32 \
  --fine-n 64 \
  --epochs 2 \
  --batch-size 1 \
  --device cpu
```

Darcy correction evaluation:
```bash
/Users/zhougc/miniconda3/envs/torch_310/bin/python scripts/eval_2d_darcy_correction.py \
  --run-dir runs/<stage1_experiment_name>_correction
```

## Visualization Notebook
Notebook path:
```bash
/Users/zhougc/Desktop/IID/LRTOR_project/Bias_Aware_FNO/output/jupyter-notebook/ns-5holes-fenicsx-dataset-visualization.ipynb
```

This notebook reads generated `.mat` files from `data/`, summarizes tensor shapes and metadata, and visualizes:
- obstacle masks and hole geometry,
- initial vorticity plus trajectory snapshots,
- per-sample galleries,
- geometry and dynamics statistics,
- optional velocity/pressure auxiliary fields when generated with `--save-aux-fields`.

Darcy notebook path:
```bash
/Users/zhougc/Desktop/IID/LRTOR_project/Bias_Aware_FNO/output/jupyter-notebook/darcy-fem-coarse-fine-visualization.ipynb
```

The Darcy notebook reads paired train/test `.mat` files, shows coarse/fine mesh metadata, visualizes `coeff`, `sol_coarse`, `sol_fine`, `error_hf_lf`, and can also compare saved predictions from a run directory.

Darcy correction notebook path:
```bash
/Users/zhougc/Desktop/IID/LRTOR_project/Bias_Aware_FNO/output/jupyter-notebook/darcy-fno-correction-visualization.ipynb
```

The correction notebook reads a correction run directory and visualizes:
- `pred_backbone`
- `pred_corrected`
- `target_fine`
- `b_h`
- `|tau_x| + |tau_y|`
- `residual_corrected`
- `flux_error_corrected`
- sample-level backbone vs corrected improvement statistics

The new default workflow is:
1. Generate data into `data/`
2. Train once
3. Inspect artifacts inside `runs/<experiment_name>/`

For new experiments, `runs/<experiment_name>/predictions.mat` and `runs/<experiment_name>/eval_metrics.json` are the primary test outputs.
