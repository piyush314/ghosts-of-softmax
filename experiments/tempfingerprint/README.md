# Temperature Fingerprint Experiment

Tests that loss-inflation curves collapse when scaled by the temperature-adjusted ratio r_T = tau * Delta_a / (pi * T).

## What it does

1. Trains MLP on digits at temperatures T in {0.5, 1.0, 2.0, 4.0}
2. Sweeps one-step sizes at trained checkpoints
3. Plots loss inflation vs r_T
4. Confirms universal curve collapse across temperatures

## Run

```bash
python run.py
```

Results are saved to `results/`.
