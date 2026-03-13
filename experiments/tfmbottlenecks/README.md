# Transformer Bottlenecks Experiment

Compares adaptive vs fixed learning rates on a tiny transformer, testing which radii (output, attention, FFN) matter most.

## What it does

1. Trains a small transformer on digits with 5 LR strategies:
   - all-radii: min(rho_out, rho_attn, rho_ffn)
   - attn-out: min(rho_out, rho_attn)
   - output-only: rho_out only
   - fixed-1x and fixed-16x baselines
2. Tracks per-component radii over training
3. Shows bottleneck shifts from FFN to attention to output

## Run

```bash
python run.py
```

Results are saved to `results/`.
