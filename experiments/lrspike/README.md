# LR Spike Experiment

Compares plain Adam vs rho-controller under a 1000x learning-rate spike.

## What it does

1. Trains MLP on digits with base LRs {1e-4, 1e-3, 1e-2}
2. Injects a 1000x spike at step 50
3. Runs both plain Adam and rho-scaled Adam
4. Shows rho-controller absorbs the spike without loss explosion

## Run

```bash
python run.py
```

Results are saved to `results/`.
