# Bias_Aware_FNO

FNO-style scripts for random five-hole Brinkman-penalized Navier-Stokes data generation, training, and evaluation.

Use the PyTorch environment on this machine:
```bash
/Users/zhougc/miniconda3/envs/torch_310/bin/python
```

## Layout
- `Adam.py`, `utilities3.py`: utilities kept in the original `fourier-neural-operator` style.
- `data_generation/navier_stokes_5holes/ns_2d_brinkman.py`: dataset generator.
- `fourier_2d_time_5holes.py`: recurrent one-step FNO training script.
- `scripts/eval_2d_time_5holes.py`: rollout evaluation script.

## Data Generation
```bash
cd /Users/zhougc/Desktop/IID/LRTOR_project/Bias_Aware_FNO
/Users/zhougc/miniconda3/envs/torch_310/bin/python data_generation/navier_stokes_5holes/ns_2d_brinkman.py
```

Smoke test:
```bash
/Users/zhougc/miniconda3/envs/torch_310/bin/python data_generation/navier_stokes_5holes/ns_2d_brinkman.py \
  --resolution 64 \
  --n-train 8 \
  --n-test 4 \
  --record-steps 10 \
  --final-time 1.0 \
  --delta-t 1e-3 \
  --batch-size 4 \
  --device cpu
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
