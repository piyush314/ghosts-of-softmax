#!/usr/bin/env python3
"""Verify reproducibility of loss-inflation phase transition near r = tau/rho_a = 1.

Protocol:
1) Train identical MLP architecture on digits for multiple random seeds.
2) Capture checkpoints near target test accuracies: early/mid/late.
3) At each checkpoint, evaluate:
   - Gradient direction loss-inflation curve vs r.
   - Random-direction median loss-inflation curve vs r.
   - Hessian step ratio tau_H/rho_a for random directions.
4) Aggregate across seeds and report onset statistics.
"""

from __future__ import annotations

import copy
import gzip
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ghosts.reporting import repo_relpath, write_summary

FIG_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [7, 11, 19, 23]
TARGETS = [0.84, 0.95, 0.98]
STAGE_NAMES = ["early", "mid", "late"]

MAX_STEPS = 750
BATCH_SIZE = 128
EVAL_N = 384
RANDOM_DIRS = 20
EVAL_INTERVAL = 10

R_VALUES = np.logspace(-3, np.log10(35.0), 42)

PALETTE = {
    "red": "#E3120B",
    "blue": "#006BA2",
    "green": "#2E8B57",
    "gray": "#D0D0D0",
    "dark": "#3D3D3D",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
    }
)


def load_digits_csv() -> Tuple[np.ndarray, np.ndarray]:
    import site
    candidates = []
    for sp in site.getsitepackages():
        candidates.append(Path(sp) / "sklearn" / "datasets" / "data" / "digits.csv.gz")
    usp = site.getusersitepackages()
    if usp:
        candidates.append(Path(usp) / "sklearn" / "datasets" / "data" / "digits.csv.gz")
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError("Could not find sklearn digits.csv.gz")
    arr = np.loadtxt(gzip.open(csv_path, "rt"), delimiter=",", dtype=np.float32)
    X = arr[:, :-1] / 16.0
    y = arr[:, -1].astype(np.int64)
    return X, y


def stratified_split(
    X: np.ndarray, y: np.ndarray, test_ratio: float = 0.30, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = int(round(test_ratio * len(idx)))
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class MLPDigits(nn.Module):
    def __init__(self, width: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(64, width)
        self.ln1 = nn.LayerNorm(width)
        self.fc2 = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.fc3 = nn.Linear(width, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.ln1(self.fc1(x)))
        x = F.gelu(self.ln2(self.fc2(x)))
        return self.fc3(x)


def model_test_acc(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        z = model(X)
        return float((z.argmax(dim=1) == y).float().mean().item())


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def train_snapshots_for_seed(
    seed: int,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> Dict[str, Dict]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 101)

    model = MLPDigits(width=128)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)

    snapshots: Dict[str, Dict] = {}
    hit = [False, False, False]
    n = len(X_train)

    for step in range(MAX_STEPS + 1):
        if step % EVAL_INTERVAL == 0:
            acc = model_test_acc(model, X_test, y_test)
            for i, t in enumerate(TARGETS):
                if (not hit[i]) and (acc >= t):
                    snapshots[STAGE_NAMES[i]] = {
                        "state": clone_state_dict(model),
                        "step": float(step),
                        "acc": float(acc),
                    }
                    hit[i] = True
            if all(hit):
                break

        if step == MAX_STEPS:
            break

        idx = rng.choice(n, size=BATCH_SIZE, replace=False)
        xb = X_train[idx]
        yb = y_train[idx]
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()

    # Fallback: if a target wasn't hit, use final model state.
    final_acc = model_test_acc(model, X_test, y_test)
    final_state = clone_state_dict(model)
    final_step = float(step)
    for i, name in enumerate(STAGE_NAMES):
        if name not in snapshots:
            snapshots[name] = {
                "state": copy.deepcopy(final_state),
                "step": final_step,
                "acc": float(final_acc),
            }
    return snapshots


def get_named_params(model: nn.Module) -> List[Tuple[str, torch.Tensor]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def grad_direction(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor
) -> Tuple[Dict[str, torch.Tensor], float]:
    named = get_named_params(model)
    params = [p for _, p in named]
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(X), y)
    grads = torch.autograd.grad(loss, params, create_graph=False)
    gnorm = float(torch.sqrt(sum((g * g).sum() for g in grads)).item())
    direction = {n: (g / max(gnorm, 1e-12)).detach().clone() for (n, _), g in zip(named, grads)}
    return direction, float(loss.item())


def random_unit_direction(model: nn.Module, rng: np.random.Generator) -> Dict[str, torch.Tensor]:
    named = get_named_params(model)
    raw: Dict[str, torch.Tensor] = {}
    sq = 0.0
    for n, p in named:
        arr = rng.standard_normal(size=tuple(p.shape)).astype(np.float32)
        t = torch.from_numpy(arr)
        raw[n] = t
        sq += float((t * t).sum().item())
    norm = math.sqrt(max(sq, 1e-12))
    return {n: raw[n] / norm for n, _ in named}


def logits_and_slopes(
    model: nn.Module, X: torch.Tensor, direction: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    params = {n: p.detach() for n, p in model.named_parameters()}

    def f(pdict):
        return functional_call(model, pdict, (X,))

    logits = f(params).detach()
    _, slopes = jvp(f, (params,), (direction,))
    return logits, slopes.detach()


def rho_from_slopes(slopes: torch.Tensor) -> float:
    delta_a = (slopes.max(dim=1).values - slopes.min(dim=1).values)
    da_max = float(delta_a.max().item())
    return math.pi / max(da_max, 1e-12)


def directional_curvature(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, direction: Dict[str, torch.Tensor]
) -> float:
    named = get_named_params(model)
    params = [p for _, p in named]
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(X), y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    dot = torch.zeros((), dtype=loss.dtype)
    for (name, _), g in zip(named, grads):
        dot = dot + (g * direction[name]).sum()
    hv = torch.autograd.grad(dot, params, create_graph=False, retain_graph=False)
    curv = torch.zeros((), dtype=loss.dtype)
    for (name, _), hv_i in zip(named, hv):
        curv = curv + (hv_i * direction[name]).sum()
    return float(curv.item())


def loss_ratio_curve(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    direction: Dict[str, torch.Tensor],
    rho: float,
    base_loss: float,
    r_values: np.ndarray,
) -> np.ndarray:
    params = {n: p.detach() for n, p in model.named_parameters()}
    out = []
    for r in r_values:
        tau = float(r * rho)
        shifted = {n: params[n] + tau * direction[n] for n in params}
        with torch.no_grad():
            z = functional_call(model, shifted, (X,))
            l = float(F.cross_entropy(z, y).item())
        out.append(l / max(base_loss, 1e-12))
    return np.array(out, dtype=np.float64)


def first_crossing(r_values: np.ndarray, y: np.ndarray, threshold: float = 1.1) -> float:
    idx = np.where(y >= threshold)[0]
    return float(r_values[idx[0]]) if len(idx) else float("nan")


def evaluate_stage(
    model: nn.Module,
    X_eval: torch.Tensor,
    y_eval: torch.Tensor,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 2222)

    g_dir, base_loss = grad_direction(model, X_eval, y_eval)
    _, g_slopes = logits_and_slopes(model, X_eval, g_dir)
    g_rho = rho_from_slopes(g_slopes)
    g_curve = loss_ratio_curve(model, X_eval, y_eval, g_dir, g_rho, base_loss, R_VALUES)

    random_curves = []
    h_ratios = []
    for _ in range(RANDOM_DIRS):
        d = random_unit_direction(model, rng)
        _, slopes = logits_and_slopes(model, X_eval, d)
        rho = rho_from_slopes(slopes)
        curve = loss_ratio_curve(model, X_eval, y_eval, d, rho, base_loss, R_VALUES)
        random_curves.append(curve)

        curv = directional_curvature(model, X_eval, y_eval, d)
        tau_h = 2.0 / max(curv, 1e-12)
        h_ratios.append(tau_h / max(rho, 1e-12))

    random_curves_arr = np.array(random_curves)
    return {
        "grad_curve": g_curve,
        "rand_curves": random_curves_arr,
        "rand_median": np.median(random_curves_arr, axis=0),
        "rand_q25": np.percentile(random_curves_arr, 25, axis=0),
        "rand_q75": np.percentile(random_curves_arr, 75, axis=0),
        "h_ratio": np.array(h_ratios, dtype=np.float64),
        "g_rho": np.array([g_rho], dtype=np.float64),
        "base_loss": np.array([base_loss], dtype=np.float64),
    }


def main() -> None:
    X, y = load_digits_csv()
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_ratio=0.30, seed=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    eval_rng = np.random.default_rng(1234)
    eval_idx = eval_rng.choice(len(X_train_t), size=min(EVAL_N, len(X_train_t)), replace=False)
    X_eval_t = X_train_t[eval_idx]
    y_eval_t = y_train_t[eval_idx]

    all_results: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for stage in STAGE_NAMES:
        all_results[stage] = {
            "grad_curve": [],
            "rand_median": [],
            "rand_q25": [],
            "rand_q75": [],
            "h_ratio_med": [],
            "onset_grad": [],
            "onset_rand": [],
            "acc": [],
            "step": [],
        }

    print("=" * 92)
    print("EXP10: REPRODUCIBILITY CHECK FOR PHASE TRANSITION (RANDOM DIRECTIONS)")
    print("=" * 92)

    for seed in SEEDS:
        print(f"\n[seed={seed}] training checkpoints...")
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = MLPDigits(width=128)
        snaps = train_snapshots_for_seed(seed, X_train_t, y_train_t, X_test_t, y_test_t)

        for stage in STAGE_NAMES:
            model.load_state_dict(snaps[stage]["state"])
            out = evaluate_stage(model, X_eval_t, y_eval_t, seed + hash(stage) % 10_000)

            onset_g = first_crossing(R_VALUES, out["grad_curve"], threshold=1.1)
            onset_r = first_crossing(R_VALUES, out["rand_median"], threshold=1.1)
            h_med = float(np.median(out["h_ratio"]))

            all_results[stage]["grad_curve"].append(out["grad_curve"])
            all_results[stage]["rand_median"].append(out["rand_median"])
            all_results[stage]["rand_q25"].append(out["rand_q25"])
            all_results[stage]["rand_q75"].append(out["rand_q75"])
            all_results[stage]["h_ratio_med"].append(np.array([h_med]))
            all_results[stage]["onset_grad"].append(np.array([onset_g]))
            all_results[stage]["onset_rand"].append(np.array([onset_r]))
            all_results[stage]["acc"].append(np.array([snaps[stage]["acc"]]))
            all_results[stage]["step"].append(np.array([snaps[stage]["step"]]))

            print(
                f"  stage={stage:<5} acc={snaps[stage]['acc']:.3f} step={int(snaps[stage]['step']):>3d} "
                f"onset_grad@1.1={onset_g:.3f} onset_rand@1.1={onset_r:.3f} hratio_med={h_med:.1f}x"
            )

    # Aggregate and plot.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharey=True)
    summary_lines = []
    for j, stage in enumerate(STAGE_NAMES):
        ax = axes[j]

        grad_curves = np.array(all_results[stage]["grad_curve"])
        rand_med_curves = np.array(all_results[stage]["rand_median"])
        h_med_arr = np.array([x.item() for x in all_results[stage]["h_ratio_med"]])
        acc_arr = np.array([x.item() for x in all_results[stage]["acc"]])

        grad_med = np.median(grad_curves, axis=0)
        rand_med = np.median(rand_med_curves, axis=0)

        # show per-seed random-median as faint gray for reproducibility texture
        for k in range(rand_med_curves.shape[0]):
            ax.plot(R_VALUES, rand_med_curves[k], color=PALETTE["gray"], lw=1.0, alpha=0.45, zorder=1)

        ax.plot(R_VALUES, grad_med, color=PALETTE["red"], lw=2.6, label="gradient", zorder=3)
        ax.plot(R_VALUES, rand_med, color=PALETTE["blue"], lw=2.6, label="median random", zorder=4)
        ax.axvline(1.0, color=PALETTE["red"], ls="--", lw=2, alpha=0.85, label="r=1")

        h_median = float(np.median(h_med_arr))
        ax.axvline(
            h_median,
            color=PALETTE["green"],
            ls=":",
            lw=2.5,
            alpha=0.9,
            label=f"2/κ ≈ {h_median:.1f}×ρ_a",
        )

        onset_grad_arr = np.array([x.item() for x in all_results[stage]["onset_grad"]])
        onset_rand_arr = np.array([x.item() for x in all_results[stage]["onset_rand"]])
        valid_rand = np.isfinite(onset_rand_arr)
        frac_near_1 = float(np.mean((onset_rand_arr[valid_rand] >= 0.8) & (onset_rand_arr[valid_rand] <= 1.25)))

        summary_lines.append(
            {
                "stage": stage,
                "acc_med": float(np.median(acc_arr)),
                "onset_grad_med": float(np.nanmedian(onset_grad_arr)),
                "onset_rand_med": float(np.nanmedian(onset_rand_arr)),
                "onset_rand_min": float(np.nanmin(onset_rand_arr)),
                "onset_rand_max": float(np.nanmax(onset_rand_arr)),
                "frac_near_1": frac_near_1,
                "h_ratio_med": h_median,
            }
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(R_VALUES.min(), R_VALUES.max())
        ax.set_ylim(0.12, 80)
        ax.set_xlabel("r = tau / rho_a")
        if j == 0:
            ax.set_ylabel("Loss ratio")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(f"{stage} (acc≈{np.median(acc_arr):.2f})", loc="center")
        ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.85)

    fig.suptitle(
        "Reproducibility Check: Loss Inflation Onset vs r = tau/rho_a (random directions, multi-seed)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig_path = FIG_DIR / "exp10_phase_repro_randomdirs.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Save compact arrays.
    npz_payload = {"r_values": R_VALUES}
    for stage in STAGE_NAMES:
        npz_payload[f"{stage}_grad_curve_seeds"] = np.array(all_results[stage]["grad_curve"])
        npz_payload[f"{stage}_rand_med_curve_seeds"] = np.array(all_results[stage]["rand_median"])
        npz_payload[f"{stage}_h_ratio_med_seeds"] = np.array(
            [x.item() for x in all_results[stage]["h_ratio_med"]]
        )
        npz_payload[f"{stage}_onset_grad_seeds"] = np.array(
            [x.item() for x in all_results[stage]["onset_grad"]]
        )
        npz_payload[f"{stage}_onset_rand_seeds"] = np.array(
            [x.item() for x in all_results[stage]["onset_rand"]]
        )
        npz_payload[f"{stage}_acc_seeds"] = np.array([x.item() for x in all_results[stage]["acc"]])
        npz_payload[f"{stage}_step_seeds"] = np.array([x.item() for x in all_results[stage]["step"]])
    npz_path = FIG_DIR / "exp10_phase_repro_randomdirs_results.npz"
    np.savez(npz_path, **npz_payload)
    summary_path = FIG_DIR / "summary.json"
    write_summary(
        summary_path,
        {
            "experiment_id": "phasetransition",
            "config": {
                "seeds": SEEDS,
                "targets": TARGETS,
                "stage_names": STAGE_NAMES,
                "eval_n": EVAL_N,
            },
            "artifacts": {
                "plot_png": repo_relpath(fig_path, REPO_ROOT),
                "results_npz": repo_relpath(npz_path, REPO_ROOT),
                "summary_json": repo_relpath(summary_path, REPO_ROOT),
            },
            "stages": summary_lines,
        },
    )

    print("\n" + "=" * 92)
    print("SUMMARY")
    print("=" * 92)
    for s in summary_lines:
        print(
            f"{s['stage']:<5} acc_med={s['acc_med']:.3f} "
            f"onset_rand_med@1.1={s['onset_rand_med']:.3f} "
            f"[{s['onset_rand_min']:.3f},{s['onset_rand_max']:.3f}] "
            f"onset_grad_med@1.1={s['onset_grad_med']:.3f} "
            f"near_r1_frac={100.0*s['frac_near_1']:.1f}% "
            f"median(2/κρ)={s['h_ratio_med']:.1f}x"
        )

    print("\nSaved:")
    print(f"  - {fig_path.resolve()}")
    print(f"  - {npz_path.resolve()}")
    print(f"  - {summary_path.resolve()}")


if __name__ == "__main__":
    main()
