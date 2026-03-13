# ResNet Natural Instability Experiment

Detects natural (non-injected) instabilities on CIFAR-10 with ResNet-18.

## What it does

1. Trains ResNet-18 on CIFAR-10 at several learning rates
2. Logs the instability ratio r = tau/rho_a at every step
3. Tests whether r approaching 1 is a leading indicator of loss spikes
4. No artificial perturbations — instabilities arise naturally

## Run

```bash
python run.py
```

Results are saved to `results/`.
