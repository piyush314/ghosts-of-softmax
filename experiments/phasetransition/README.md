# Phase Transition Experiment

Verifies the loss-inflation phase transition near r = tau/rho_a = 1 across multiple seeds and training stages.

## What it does

1. Trains an MLP on the digits dataset to early/mid/late accuracy checkpoints
2. At each checkpoint, sweeps step sizes along gradient and random directions
3. Measures loss inflation as a function of r = tau/rho_a
4. Confirms the phase transition: r < 1 safe, r > 1 inflates

## Run

```bash
python run.py            # full run
python plot.py           # loss-inflation curves
python plotjvp.py        # JVP-based radius plots
python plotsigmoid.py    # sigmoid fit to transition
```

Results are saved to `results/`.
