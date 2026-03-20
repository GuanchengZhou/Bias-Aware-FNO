# Bias_Aware_FNO

FNO-style training and evaluation scripts for random five-hole Navier-Stokes data.

Data generation now uses `gmsh + dolfinx + PETSc` in `fenicsx-env`.
Training and evaluation still use the original FNO-style PyTorch scripts in `torch_310`.

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

## Data Generation
```bash
cd /Users/zhougc/Desktop/IID/LRTOR_project/Bias_Aware_FNO
/Users/zhougc/miniconda3/envs/fenicsx-env/bin/python data_generation/navier_stokes_5holes/ns_2d_fenicsx.py
```

Large datasets automatically fall back to an HDF5-backed `.mat` file when classic MAT v5 would exceed the 4 GB array limit. The existing training scripts and visualization notebook can read both formats.

Smoke test:
```bash
/Users/zhougc/miniconda3/envs/fenicsx-env/bin/python data_generation/navier_stokes_5holes/ns_2d_fenicsx.py \
  --grid-resolution 64 \
  --n-train 2 \
  --n-test 1 \
  --record-steps 10 \
  --final-time 1.0 \
  --dt 0.1 \
  --mesh-size-min 0.10 \
  --mesh-size-max 0.14
```

Legacy spectral baseline:
```bash
/Users/zhougc/miniconda3/envs/torch_310/bin/python data_generation/navier_stokes_5holes/ns_2d_brinkman.py
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

## Evaluation
```bash
/Users/zhougc/miniconda3/envs/torch_310/bin/python scripts/eval_2d_time_5holes.py
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
