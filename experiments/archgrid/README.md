# Architecture Grid Experiment

Multi-seed comparison of Transformer, MLP+LN, and CNN+BN under a 10x LR spike.

## What it does

1. Trains three architectures on digits with multiple seeds
2. Injects a 10x spike at step 50
3. Compares plain Adam vs rho-controller
4. Generates a grid of loss curves showing controller robustness across architectures

## Run

```bash
python run.py
```

Results are saved to `results/`.
