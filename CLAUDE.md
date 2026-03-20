# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bias-Aware FNO: Fourier Neural Operator training/evaluation for 2D Navier-Stokes flow through random five-hole geometries. The project has two distinct phases with separate conda environments:

1. **Data generation** (FEniCSx): `gmsh + dolfinx + PETSc` solves Navier-Stokes on unstructured meshes, interpolates to regular grids, saves `.mat` files.
2. **Training/evaluation** (PyTorch): Recurrent one-step FNO predicts vorticity rollouts from initial conditions + obstacle masks.

## Environments

- **PyTorch** (`torch_310`): `/Users/zhougc/miniconda3/envs/torch_310/bin/python`
- **FEniCSx** (`fenicsx-env`): `/Users/zhougc/miniconda3/envs/fenicsx-env/bin/python`

## Key Commands

### Data generation (uses fenicsx-env)
```bash
# Full run
/Users/zhougc/miniconda3/envs/fenicsx-env/bin/python data_generation/navier_stokes_5holes/ns_2d_fenicsx.py

# Smoke test
/Users/zhougc/miniconda3/envs/fenicsx-env/bin/python data_generation/navier_stokes_5holes/ns_2d_fenicsx.py \
  --grid-resolution 64 --n-train 2 --n-test 1 --record-steps 10 --final-time 1.0 --dt 0.1 --mesh-density 10
```

### Training (uses torch_310)
```bash
# Full run
/Users/zhougc/miniconda3/envs/torch_310/bin/python fourier_2d_time_5holes.py
