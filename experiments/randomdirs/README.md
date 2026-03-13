# Random Directions Experiment

Validates rho_a along random parameter directions using exact JVP.

## What it does

1. Trains a linear classifier on digits
2. Samples random parameter directions
3. Computes exact rho_a via JVP for each direction
4. Sweeps step sizes and measures loss inflation
5. Confirms phase transition holds for arbitrary (non-gradient) directions

## Run

```bash
python run.py    # compute sweeps
python plot.py   # generate figures
```

Results are saved to `results/`.
