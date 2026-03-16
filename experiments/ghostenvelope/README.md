# Ghost Envelope Experiment

Visualizes the complex ghost locations for several small architectures across
training, together with quantile envelope bands and the corresponding
`rho_a` level.

## What it does

1. Trains multiple architectures on digits
2. Snapshots ghost geometry at untrained, mid-training, and final stages
3. Saves the raw ghost statistics and renders the envelope figure
4. Writes a compact machine-readable summary for reproducibility checks

## Run

```bash
python run.py
```

Results are saved to `cache/` and `results/`.
